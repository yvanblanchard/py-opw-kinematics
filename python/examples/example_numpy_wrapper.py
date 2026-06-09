"""
example_numpy_wrapper.py — Runnable test/demo for numpy_wrapper.py.

Shows all main Rust library methods through the numpy wrapper:
  - KinematicModel construction
  - Forward kinematics (single & batch)
  - Inverse kinematics (single & batch)
  - Forward link frames
  - FK → IK roundtrip validation
  - URDF loading (file and string, commented out)

Run directly:
    python example_numpy_wrapper.py

Or via pytest (all test_* functions are collected automatically):
    pytest example_numpy_wrapper.py -v
"""

import numpy as np

from py_opw_kinematics.numpy_wrapper import (
    KinematicModel,
    Robot,
    from_urdf_string,
)


# ---------------------------------------------------------------------------
# Shared robot fixture (Comau-like geometry, same as existing tests)
# ---------------------------------------------------------------------------

def make_robot(degrees: bool = True) -> Robot:
    km = KinematicModel(
        a1=400.333,
        a2=-251.449,
        b=0.0,
        c1=830.0,
        c2=1177.556,
        c3=1443.593,
        c4=230.0,
        offsets=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        flip_axes=(True, False, True, True, False, True),
        has_parallelogram=True,
    )
    return Robot(km, degrees=degrees)


JOINTS_HOME = [0.0, 0.0, -90.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# 1. KinematicModel — construction and attribute access
# ---------------------------------------------------------------------------

def test_kinematic_model_attributes():
    km = KinematicModel(a1=1.0, c1=2.0, c2=3.0, c3=4.0, c4=5.0)
    assert km.a1 == 1.0
    assert km.c1 == 2.0
    assert not km.has_parallelogram
    assert len(km.offsets) == 6
    assert len(km.flip_axes) == 6
    print("PASS  test_kinematic_model_attributes")


# ---------------------------------------------------------------------------
# 2. Robot repr
# ---------------------------------------------------------------------------

def test_robot_repr():
    robot = make_robot()
    r = repr(robot)
    assert "Robot" in r and "KinematicModel" in r
    print("PASS  test_robot_repr")


# ---------------------------------------------------------------------------
# 3. Forward kinematics — known position check
# ---------------------------------------------------------------------------

def test_forward_kinematics_position():
    robot = make_robot(degrees=True)
    pose = robot.forward(JOINTS_HOME)

    assert pose.shape == (4, 4)
    # Known TCP X, Y, Z for this robot at home (no EE transform)
    translation = pose[:3, 3]
    expected = np.array([2073.926, 0.0, 2259.005])
    assert np.allclose(translation, expected, atol=1.0), (
        f"Position mismatch: {translation} vs {expected}"
    )
    print(f"PASS  test_forward_kinematics_position  TCP={translation.round(3)}")


# ---------------------------------------------------------------------------
# 4. Forward kinematics — with end-effector transform
# ---------------------------------------------------------------------------

def test_forward_kinematics_with_ee():
    robot = make_robot(degrees=True)

    # 90-degree rotation around Y as EE offset
    ee = np.array([
        [ 0.0, 0.0, 1.0, 0.0],
        [ 0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    pose = robot.forward(JOINTS_HOME, ee_transform=ee)
    assert pose.shape == (4, 4)
    assert pose[3, 3] == 1.0          # homogeneous row
    print(f"PASS  test_forward_kinematics_with_ee  TCP={pose[:3,3].round(3)}")


# ---------------------------------------------------------------------------
# 5. Inverse kinematics — recovers original joints
# ---------------------------------------------------------------------------

def test_inverse_kinematics():
    robot = make_robot(degrees=True)
    joints_in = np.array([10.0, 20.0, -90.0, 30.0, 20.0, 10.0])

    pose = robot.forward(joints_in)
    solutions = robot.inverse(pose, current_joints=joints_in)

    assert len(solutions) > 0, "Expected at least one IK solution"
    found = any(np.allclose(sol, joints_in, atol=1e-2) for sol in solutions)
    assert found, f"Original joints not in solutions: {solutions}"
    print(f"PASS  test_inverse_kinematics  ({len(solutions)} solutions)")


# ---------------------------------------------------------------------------
# 6. FK → IK roundtrip for radians mode
# ---------------------------------------------------------------------------

def test_fk_ik_roundtrip_radians():
    robot = make_robot(degrees=False)
    joints_in = np.deg2rad([0.0, 0.0, -90.0, 0.0, 0.0, 0.0])

    pose = robot.forward(joints_in)
    solutions = robot.inverse(pose, current_joints=joints_in)

    assert len(solutions) > 0
    # Verify each solution reproduces the same TCP pose
    for sol in solutions:
        pose2 = robot.forward(sol)
        assert np.allclose(pose[:3, 3], pose2[:3, 3], atol=1e-3)
        assert np.allclose(pose[:3, :3], pose2[:3, :3], atol=1e-3)
    print("PASS  test_fk_ik_roundtrip_radians")


# ---------------------------------------------------------------------------
# 7. Forward link frames
# ---------------------------------------------------------------------------

def test_forward_frames():
    robot = make_robot(degrees=True)
    frames = robot.forward_frames(JOINTS_HOME)

    # [Base, J1, J2, J3, J4, J5, J6, TCP] = 8 frames
    assert len(frames) == 8, f"Expected 8 frames, got {len(frames)}"
    for i, f in enumerate(frames):
        assert f.shape == (4, 4), f"Frame {i} has wrong shape {f.shape}"
        assert np.allclose(f[3], [0, 0, 0, 1], atol=1e-9), \
            f"Frame {i} bottom row is not [0,0,0,1]"
    # TCP (last frame) should match robot.forward()
    tcp_from_frames = frames[-1]
    tcp_from_forward = robot.forward(JOINTS_HOME)
    assert np.allclose(tcp_from_frames, tcp_from_forward, atol=1e-3)
    print(f"PASS  test_forward_frames  ({len(frames)} frames)")


# ---------------------------------------------------------------------------
# 8. Batch forward kinematics
# ---------------------------------------------------------------------------

def test_batch_forward():
    robot = make_robot(degrees=True)
    np.random.seed(0)
    joints_batch = np.column_stack([
        np.random.uniform(-180, 180, 20),
        np.random.uniform(-90, 90, 20),
        np.random.uniform(-180, 180, 20),
        np.random.uniform(-180, 180, 20),
        np.random.uniform(-90, 90, 20),
        np.random.uniform(-180, 180, 20),
    ])

    poses = robot.batch_forward(joints_batch)

    assert poses.shape == (20, 4, 4)
    # Spot-check: first row should match single forward
    expected_first = robot.forward(joints_batch[0])
    assert np.allclose(poses[0], expected_first, atol=1e-9)
    print(f"PASS  test_batch_forward  shape={poses.shape}")


# ---------------------------------------------------------------------------
# 9. Batch inverse kinematics
# ---------------------------------------------------------------------------

def test_batch_inverse():
    robot = make_robot(degrees=True)
    joints_in = np.array([
        [0.0,  0.0, -90.0,  0.0,  0.0,  0.0],
        [10.0, 0.0, -90.0,  0.0,  0.0,  0.0],
        [10.0,10.0, -90.0,  0.0,  0.0,  0.0],
    ])

    poses_4x4 = robot.batch_forward(joints_in)         # (3, 4, 4)
    result = robot.batch_inverse(poses_4x4, current_joints=joints_in[0])

    assert result.shape == (3, 6), f"Expected (3,6), got {result.shape}"
    print(f"PASS  test_batch_inverse  shape={result.shape}")


# ---------------------------------------------------------------------------
# 10. Batch FK → IK roundtrip
# ---------------------------------------------------------------------------

def test_batch_roundtrip():
    robot = make_robot(degrees=True)
    joints_in = np.column_stack([
        np.linspace(-90, 90, 10),
        np.linspace(-45, 45, 10),
        np.linspace(-90, 90, 10),
        np.linspace(-90, 90, 10),
        np.linspace(-45, 45, 10),
        np.linspace(-90, 90, 10),
    ])

    poses = robot.batch_forward(joints_in)
    result = robot.batch_inverse(poses, current_joints=joints_in[0])

    valid = ~np.isnan(result).any(axis=1)
    assert valid.all(), f"Some poses had no IK solution: {np.where(~valid)}"
    assert np.allclose(joints_in, result, atol=1e-3), (
        f"Roundtrip mismatch:\n{joints_in}\nvs\n{result}"
    )
    print(f"PASS  test_batch_roundtrip  ({valid.sum()}/{len(valid)} valid)")


# ---------------------------------------------------------------------------
# 11. URDF loading (file)  — uncomment and adjust path to run
# ---------------------------------------------------------------------------

# def test_from_urdf_file():
#     km = from_urdf_file("path/to/robot.urdf")
#     assert km.c1 > 0
#     robot = Robot(km)
#     pose = robot.forward([0, 0, 0, 0, 0, 0])
#     assert pose.shape == (4, 4)
#     print("PASS  test_from_urdf_file")


# ---------------------------------------------------------------------------
# 12. URDF loading (string)
# ---------------------------------------------------------------------------

def test_from_urdf_string_minimal():
    """
    Minimal URDF with 6 revolute joints — just checks that parsing doesn't crash
    and returns a KinematicModel with numeric DH parameters.
    """
    urdf_xml = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link"/>
  <link name="link1"/>
  <link name="link2"/>
  <link name="link3"/>
  <link name="link4"/>
  <link name="link5"/>
  <link name="link6"/>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/><child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>
  <joint name="joint3" type="revolute">
    <parent link="link2"/><child link="link3"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
  </joint>
  <joint name="joint4" type="revolute">
    <parent link="link3"/><child link="link4"/>
    <origin xyz="0.05 0 0.05" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
  </joint>
  <joint name="joint5" type="revolute">
    <parent link="link4"/><child link="link5"/>
    <origin xyz="0.2 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>
  <joint name="joint6" type="revolute">
    <parent link="link5"/><child link="link6"/>
    <origin xyz="0.05 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
  </joint>
</robot>"""
    km = from_urdf_string(urdf_xml)
    assert isinstance(km, KinematicModel)
    robot = Robot(km, degrees=True)
    pose = robot.forward([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert pose.shape == (4, 4)
    print(f"PASS  test_from_urdf_string_minimal  TCP={pose[:3,3].round(4)}")


# ---------------------------------------------------------------------------
# Entry point — run all tests when executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_kinematic_model_attributes,
        test_robot_repr,
        test_forward_kinematics_position,
        test_forward_kinematics_with_ee,
        test_inverse_kinematics,
        test_fk_ik_roundtrip_radians,
        test_forward_frames,
        test_batch_forward,
        test_batch_inverse,
        test_batch_roundtrip,
        test_from_urdf_string_minimal,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as exc:
            print(f"FAIL  {t.__name__}  →  {exc}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
