//! moveJ / PTP planner with joint-limit violation detection.
//!
//! Given two Cartesian targets, picks IK solutions on the same kinematic branch as the
//! supplied previous joint configuration (via `inverse_continuing`), linearly interpolates
//! in joint space, and optionally checks every sample against joint limits.

use rs_opw_kinematics::constraints::Constraints;
use rs_opw_kinematics::kinematic_traits::{Joints, Kinematics, Pose};

/// Result of planning a moveJ/PTP motion between two Cartesian targets.
pub struct MoveJPlan {
    /// Start-pose joint configuration (on the branch closest to `previous`).
    pub q_start: Joints,
    /// End-pose joint configuration (on the branch closest to `q_start`).
    pub q_end: Joints,
    /// Joint samples from start to end, inclusive (`steps + 1` entries).
    pub path: Vec<Joints>,
    /// Index of the first sample violating joint limits, if `constraints` was provided.
    pub violation_at: Option<usize>,
    /// Per-joint absolute travel between endpoints. Large values (e.g. > π/2) suggest a
    /// kinematic-branch switch between the two IK solutions.
    pub joint_deltas: Joints,
}

/// Why planning failed.
#[derive(Debug)]
pub enum MoveJError {
    /// The start pose has no reachable IK solution.
    StartUnreachable,
    /// The end pose has no reachable IK solution on (or near) the start branch.
    EndUnreachable,
    /// `steps` must be >= 1.
    InvalidSteps,
}

/// Plan a joint-interpolated motion (moveJ / PTP) between two Cartesian targets.
///
/// * `previous` seeds the start-pose IK — use the robot's current joints to preserve branch.
///   Pass `CONSTRAINT_CENTERED` if there is no previous configuration.
/// * `constraints`: when `Some`, every interpolated sample is checked with
///   `Constraints::compliant`; the first offending index is reported in `violation_at`.
///   Endpoints are also checked — an endpoint violation shows up as `violation_at = Some(0)`
///   or `Some(steps)`.
/// * `steps`: number of intervals; `steps + 1` samples are produced.
pub fn plan_move_j<K: Kinematics + ?Sized>(
    kin: &K,
    start: &Pose,
    end: &Pose,
    previous: &Joints,
    constraints: Option<&Constraints>,
    steps: usize,
) -> Result<MoveJPlan, MoveJError> {
    if steps == 0 {
        return Err(MoveJError::InvalidSteps);
    }
    let q_start = *kin
        .inverse_continuing(start, previous)
        .first()
        .ok_or(MoveJError::StartUnreachable)?;
    plan_move_j_from_joints(kin, &q_start, end, constraints, steps)
}

/// Same as [`plan_move_j`] but takes the start joint configuration directly — useful
/// when the robot's current joints are known (e.g. read from the controller) and you
/// want to avoid re-solving IK for the start pose.
pub fn plan_move_j_from_joints<K: Kinematics + ?Sized>(
    kin: &K,
    q_start: &Joints,
    end: &Pose,
    constraints: Option<&Constraints>,
    steps: usize,
) -> Result<MoveJPlan, MoveJError> {
    if steps == 0 {
        return Err(MoveJError::InvalidSteps);
    }
    let q_end = *kin
        .inverse_continuing(end, q_start)
        .first()
        .ok_or(MoveJError::EndUnreachable)?;
    Ok(interpolate_and_check(*q_start, q_end, constraints, steps))
}

fn interpolate_and_check(
    q_start: Joints,
    q_end: Joints,
    constraints: Option<&Constraints>,
    steps: usize,
) -> MoveJPlan {
    let mut joint_deltas = [0.0_f64; 6];
    for j in 0..6 {
        joint_deltas[j] = (q_end[j] - q_start[j]).abs();
    }
    let mut path = Vec::with_capacity(steps + 1);
    let mut violation_at = None;
    for i in 0..=steps {
        let t = i as f64 / steps as f64;
        let mut q = [0.0_f64; 6];
        for j in 0..6 {
            q[j] = q_start[j] + t * (q_end[j] - q_start[j]);
        }
        if violation_at.is_none() {
            if let Some(c) = constraints {
                if !c.compliant(&q) {
                    violation_at = Some(i);
                }
            }
        }
        path.push(q);
    }
    MoveJPlan {
        q_start,
        q_end,
        path,
        violation_at,
        joint_deltas,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rs_opw_kinematics::constraints::{Constraints, BY_PREV};
    use rs_opw_kinematics::kinematics_impl::OPWKinematics;
    use rs_opw_kinematics::parameters::opw_kinematics::Parameters;
    use rs_opw_kinematics::tool::Tool;
    use nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion};
    use std::sync::Arc;

    fn irb7600(base_rotation: f64) -> OPWKinematics {
        // OPW parameters for ABB IRB 7600 supplied by the user:
        //   offsets       = [0, 0, -pi/2, 0, 0, 0]
        //   flip_axes     = [true, false, true, false, true, true]  → sign_corrections
        let mut params = Parameters {
            a1: 0.410,
            a2: -0.165,
            b: 0.000,
            c1: 0.780,
            c2: 1.075,
            c3: 1.556,
            c4: 0.250,
            offsets: [0.0, 0.0, -std::f64::consts::PI / 2.0, 0.0, 0.0, 0.0],
            sign_corrections: [-1, 1, -1, 1, -1, -1],
            ..Parameters::new()
        };
        params.offsets[0] += base_rotation;
        OPWKinematics::new(params)
    }

    // Wrap a robot with a 530 mm tool offset along the flange Z axis (end-effector TCP).
    fn with_tcp(robot: OPWKinematics) -> Tool {
        Tool {
            robot: Arc::new(robot),
            tool: Isometry3::from_parts(
                Translation3::new(0.0, 0.0, 0.530),
                UnitQuaternion::identity(),
            ),
        }
    }

    // ABB quaternion convention [q1=w, q2=x, q3=y, q4=z] (scalar first).
    fn q_abb(w: f64, x: f64, y: f64, z: f64) -> UnitQuaternion<f64> {
        UnitQuaternion::from_quaternion(Quaternion::new(w, x, y, z))
    }

    // Robot base origin in the world frame (mm).
    const BASE_ORIGIN_MM: [f64; 3] = [3500.0, 1000.0, 0.0];

    // Build a pose from world-frame mm coordinates, expressed in the robot base frame.
    fn pose_mm(x: f64, y: f64, z: f64, rot: UnitQuaternion<f64>) -> Pose {
        Isometry3::from_parts(
            Translation3::new(
                (x - BASE_ORIGIN_MM[0]) * 1e-3,
                (y - BASE_ORIGIN_MM[1]) * 1e-3,
                (z - BASE_ORIGIN_MM[2]) * 1e-3,
            ),
            rot,
        )
    }

    #[test]
    fn move_j_detects_joint_limit_violation_between_two_targets() {
        // 530 mm TCP offset along the flange Z axis via Tool.
        let kin = with_tcp(irb7600(0.0));

        // Standard ABB IRB 7600-325 joint limits.
        let constraints = Constraints::from_degrees(
            [
                -180.0..=180.0,
                -60.0..=85.0,
                -180.0..=60.0,
                -300.0..=300.0,
                -100.0..=100.0,
                -220.0..=220.0,
            ],
            BY_PREV,
        );

        // Measured joint values at target 1 on the real robot (degrees).
        let q_start: Joints = [
            33.0_f64.to_radians(),
            -1.0_f64.to_radians(),
            22.0_f64.to_radians(),
            0.0,
            68.0_f64.to_radians(),
            -87.0_f64.to_radians(),
        ];

        // Target 2 in the WORLD frame (mm). pose_mm subtracts the robot base origin.
        let p2 = pose_mm(
            3226.61,
            1224.48,
            150.00,
            q_abb(0.000000, -0.342110, 0.939660, 0.000000),
        );

        // FK sanity check against target 1.
        let p1_world = [4505.88_f64, 1353.40, 0.00];
        let fk_tcp = kin.forward(&q_start).translation.vector;
        let fk_world_mm = [
            fk_tcp.x * 1e3 + BASE_ORIGIN_MM[0],
            fk_tcp.y * 1e3 + BASE_ORIGIN_MM[1],
            fk_tcp.z * 1e3 + BASE_ORIGIN_MM[2],
        ];
        println!(
            "FK(q_start) world (mm) = [{:.2}, {:.2}, {:.2}]  (expected target 1 = [{:.2}, {:.2}, {:.2}])",
            fk_world_mm[0], fk_world_mm[1], fk_world_mm[2],
            p1_world[0], p1_world[1], p1_world[2]
        );

        // Enumerate every IK branch at target 2, interpolate from q_start, and report.
        let branches = kin.inverse(&p2);
        println!("Target 2: {} IK branch(es).", branches.len());
        let mut any_violation = false;
        for (k, q_end) in branches.iter().enumerate() {
            let plan = interpolate_and_check(q_start, *q_end, Some(&constraints), 50);
            println!(
                "  branch {k}: q_end (deg) = {:?}  violation_at = {:?}  deltas (deg) = {:?}",
                q_end.map(f64::to_degrees),
                plan.violation_at,
                plan.joint_deltas.map(f64::to_degrees),
            );
            if plan.violation_at.is_some() {
                any_violation = true;
            }
        }

        // Closest-branch (controller-like) plan.
        let plan = plan_move_j_from_joints(&kin, &q_start, &p2, Some(&constraints), 50)
            .expect("IK must resolve for target 2");
        println!(
            "closest-branch plan: q_end (deg) = {:?}  deltas (deg) = {:?}  violation_at = {:?}",
            plan.q_end.map(f64::to_degrees),
            plan.joint_deltas.map(f64::to_degrees),
            plan.violation_at,
        );

        assert!(
            any_violation,
            "expected at least one IK branch at target 2 to cause a joint-limit violation on the moveJ path"
        );
    }
}

