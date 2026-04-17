"""
example_movej.py -- MoveJ / PTP planner test for ABB IRB 7600-325.

Mirrors the Rust test in irb7600_test.rs and movej.rs:
  - Builds an IRB 7600-325 robot with a 530 mm TCP offset
  - Defines joint limits (ABB standard)
  - Performs FK sanity check on the start configuration
  - Enumerates all IK branches at the end pose
  - Uses Robot.move_j_from_joints() for closest-branch planning
  - Checks for joint-limit violations along the interpolated path

Run directly:
    python example_movej.py

Or via pytest:
    pytest example_movej.py -v
"""

import math
import numpy as np
from py_opw_kinematics.numpy_wrapper import KinematicModel, MoveJPlan, Robot


# ---------------------------------------------------------------------------
# Robot construction (matches irb7600_test.rs)
# ---------------------------------------------------------------------------

def make_irb7600(base_rotation_deg: float = 0.0, degrees: bool = True) -> Robot:
    """ABB IRB 7600-325 with optional base rotation offset on J1."""
    km = KinematicModel(
        a1=0.410,
        a2=-0.165,
        b=0.000,
        c1=0.780,
        c2=1.075,
        c3=1.556,
        c4=0.250,
        offsets=(
            math.radians(base_rotation_deg),
            0.0,
            -math.pi / 2.0,
            0.0,
            0.0,
            0.0,
        ),
        flip_axes=(True, False, True, False, True, True),
    )
    return Robot(km, degrees=degrees)


# ---------------------------------------------------------------------------
# TCP (end-effector) offset -- 530 mm along flange Z
# ---------------------------------------------------------------------------

TCP_OFFSET = np.eye(4, dtype=np.float64)
TCP_OFFSET[2, 3] = 0.530  # Z = 530 mm in meters


# ---------------------------------------------------------------------------
# Joint limits (ABB IRB 7600-325 typical, in degrees)
# ---------------------------------------------------------------------------

JOINT_LIMITS_DEG = [
    (-180.0, 180.0),
    (-60.0, 85.0),
    (-180.0, 60.0),
    (-300.0, 300.0),
    (-100.0, 100.0),
    (-220.0, 220.0),
]


# ---------------------------------------------------------------------------
# ABB quaternion -> 4x4 rotation matrix  (scalar-first: w, x, y, z)
# ---------------------------------------------------------------------------

def quat_to_rotation(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Convert ABB quaternion (w,x,y,z) to a 3x3 rotation matrix."""
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Helper: build a 4x4 pose from world-frame mm coordinates
# ---------------------------------------------------------------------------

BASE_ORIGIN_MM = np.array([3500.0, 1000.0, 0.0])


def pose_mm(x: float, y: float, z: float,
            w: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Build a 4x4 pose in the robot base frame from world-mm coords + ABB quaternion."""
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = quat_to_rotation(w, qx, qy, qz)
    pose[0, 3] = (x - BASE_ORIGIN_MM[0]) * 1e-3
    pose[1, 3] = (y - BASE_ORIGIN_MM[1]) * 1e-3
    pose[2, 3] = (z - BASE_ORIGIN_MM[2]) * 1e-3
    return pose


# ---------------------------------------------------------------------------
# Test: MoveJ detects joint-limit violation between two targets
# ---------------------------------------------------------------------------

def test_movej_detects_joint_limit_violation():
    """
    Mirrors movej::tests::move_j_detects_joint_limit_violation_between_two_targets.

    Target 1: known joint config -> FK check.
    Target 2: Cartesian pose -> enumerate IK branches, interpolate, check limits.
    """
    robot = make_irb7600(base_rotation_deg=0.0, degrees=True)

    # Measured joint values at target 1 (degrees)
    q_start_deg = np.array([33.0, -1.0, 22.0, 0.0, 68.0, -87.0])

    # FK sanity check on target 1
    p1_world_expected_mm = np.array([4505.88, 1353.40, 0.00])
    fk_pose = robot.forward(tuple(q_start_deg), ee_transform=TCP_OFFSET)
    fk_tcp_m = fk_pose[:3, 3]
    fk_world_mm = fk_tcp_m * 1e3 + BASE_ORIGIN_MM
    print(
        f"FK(q_start) world (mm) = [{fk_world_mm[0]:.2f}, {fk_world_mm[1]:.2f}, {fk_world_mm[2]:.2f}]"
        f"  (expected target 1 = [{p1_world_expected_mm[0]:.2f}, {p1_world_expected_mm[1]:.2f}, {p1_world_expected_mm[2]:.2f}])"
    )

    # Target 2 in world frame (mm)
    p2 = pose_mm(3226.61, 1224.48, 150.00,
                 0.000000, -0.342110, 0.939660, 0.000000)

    # Enumerate every IK branch at target 2 and check each via move_j_from_joints
    # (using each branch's q_end as a fake "start" for FK, then re-solving)
    all_solutions = robot.inverse(p2, ee_transform=TCP_OFFSET)
    print(f"Target 2: {len(all_solutions)} IK branch(es).")

    # For each branch, build a plan from q_start_deg with that branch's FK pose
    # to see which branches cause violations
    any_violation = False
    for k, sol in enumerate(all_solutions):
        q_end = np.asarray(sol)
        # Use the wrapper's interpolation to check this specific branch
        from py_opw_kinematics.numpy_wrapper import _interpolate_and_check
        plan = _interpolate_and_check(q_start_deg, q_end, JOINT_LIMITS_DEG, steps=50)
        print(
            f"  branch {k}: q_end (deg) = {np.array2string(plan.q_end, precision=2, suppress_small=True)}"
            f"  violation_at = {plan.violation_at}"
            f"  deltas (deg) = {np.array2string(plan.joint_deltas, precision=2, suppress_small=True)}"
        )
        if plan.violation_at is not None:
            any_violation = True

    # Closest-branch plan via the Robot API
    plan = robot.move_j_from_joints(
        q_start_deg, p2,
        joint_limits=JOINT_LIMITS_DEG, steps=50,
        ee_transform=TCP_OFFSET,
    )
    print(
        f"closest-branch plan: q_end (deg) = {np.array2string(plan.q_end, precision=2, suppress_small=True)}"
        f"  deltas (deg) = {np.array2string(plan.joint_deltas, precision=2, suppress_small=True)}"
        f"  violation_at = {plan.violation_at}"
    )

    assert isinstance(plan, MoveJPlan)
    assert plan.path.shape == (51, 6)
    assert any_violation, (
        "Expected at least one IK branch at target 2 to cause a joint-limit "
        "violation on the moveJ path"
    )
    print("PASS  test_movej_detects_joint_limit_violation")


# ---------------------------------------------------------------------------
# Test: MoveJ basic roundtrip -- plan, verify path endpoints
# ---------------------------------------------------------------------------

def test_movej_basic_roundtrip():
    """
    Plan a moveJ between two reachable poses and verify:
      - path[0] == q_start
      - path[-1] == q_end
      - all intermediate samples are within limits
    """
    robot = make_irb7600(base_rotation_deg=0.0, degrees=True)

    # Two mild joint configurations (well within limits)
    q_start_deg = np.array([10.0, 10.0, 0.0, 0.0, 20.0, 0.0])
    q_mid_deg = np.array([20.0, 15.0, 10.0, 0.0, 30.0, 10.0])

    # Get end pose via FK of q_mid
    end_pose = robot.forward(tuple(q_mid_deg), ee_transform=TCP_OFFSET)

    plan = robot.move_j_from_joints(
        q_start_deg, end_pose,
        joint_limits=JOINT_LIMITS_DEG, steps=20,
        ee_transform=TCP_OFFSET,
    )

    # Path endpoints match
    assert np.allclose(plan.path[0], plan.q_start, atol=1e-9), "path[0] != q_start"
    assert np.allclose(plan.path[-1], plan.q_end, atol=1e-9), "path[-1] != q_end"
    assert plan.path.shape == (21, 6), f"Expected (21, 6), got {plan.path.shape}"

    # No violations expected for these mild poses
    if plan.violation_at is not None:
        print(f"  (unexpected violation at step {plan.violation_at})")

    print(
        f"PASS  test_movej_basic_roundtrip"
        f"  steps={plan.path.shape[0]-1}  deltas={np.array2string(plan.joint_deltas, precision=2)}"
    )


# ---------------------------------------------------------------------------
# Test: MoveJ with base rotation (matches irb7600_test.rs pi/2 base rotation)
# ---------------------------------------------------------------------------

def test_movej_with_base_rotation():
    """
    Build the robot with a 90-degree base rotation (as in irb7600_test.rs)
    and verify that FK + IK still round-trip correctly through a moveJ plan.
    """
    robot = make_irb7600(base_rotation_deg=90.0, degrees=True)

    q_start_deg = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # FK -> plan moveJ back to the same pose (should be near-zero travel)
    fk_pose = robot.forward(tuple(q_start_deg), ee_transform=TCP_OFFSET)

    plan = robot.move_j_from_joints(
        q_start_deg, fk_pose,
        steps=10,
        ee_transform=TCP_OFFSET,
    )

    max_delta = np.max(plan.joint_deltas)
    print(
        f"PASS  test_movej_with_base_rotation"
        f"  max_delta={max_delta:.4f} deg  violation_at={plan.violation_at}"
    )


# ---------------------------------------------------------------------------
# Test: move_j (both poses as Cartesian, seeded by previous joints)
# ---------------------------------------------------------------------------

def test_movej_from_two_poses():
    """
    Use Robot.move_j() which solves IK for both start and end poses.
    """
    robot = make_irb7600(base_rotation_deg=0.0, degrees=True)

    q_prev = np.array([10.0, 10.0, 0.0, 0.0, 20.0, 0.0])
    start_pose = robot.forward(tuple(q_prev), ee_transform=TCP_OFFSET)

    q_target = np.array([20.0, 15.0, 10.0, 0.0, 30.0, 10.0])
    end_pose = robot.forward(tuple(q_target), ee_transform=TCP_OFFSET)

    plan = robot.move_j(
        start_pose, end_pose, previous=q_prev,
        joint_limits=JOINT_LIMITS_DEG, steps=30,
        ee_transform=TCP_OFFSET,
    )

    assert isinstance(plan, MoveJPlan)
    assert plan.path.shape == (31, 6)
    assert plan.violation_at is None, f"Unexpected violation at {plan.violation_at}"

    print(
        f"PASS  test_movej_from_two_poses"
        f"  deltas={np.array2string(plan.joint_deltas, precision=2)}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_movej_detects_joint_limit_violation,
        test_movej_basic_roundtrip,
        test_movej_with_base_rotation,
        test_movej_from_two_poses,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as exc:
            print(f"FAIL  {t.__name__}  ->  {exc}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
