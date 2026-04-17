"""
numpy_wrapper.py — Thin Python wrapper over the py_opw_kinematics Rust extension.

Uses only NumPy (no scipy). Poses are plain 4×4 NumPy matrices.
This is a lighter alternative to __init__.py for environments without scipy.

Public API
----------
KinematicModel  — Robot geometry parameters (re-exported from _internal).
Robot           — Forward / inverse kinematics with numpy poses.
from_urdf_file  — Load KinematicModel from a URDF file path.
from_urdf_string — Load KinematicModel from URDF XML string.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ._internal import KinematicModel
from ._internal import Robot as _RobotInternal
from ._internal import from_urdf_file, from_urdf_string  # re-export

__all__ = [
    "KinematicModel",
    "MoveJPlan",
    "Robot",
    "from_urdf_file",
    "from_urdf_string",
]

# Type aliases
Joints6 = Union[Sequence[float], np.ndarray]   # length-6 joint angles
Pose4x4 = np.ndarray                            # shape (4, 4), float64
JointLimits = Sequence[Tuple[float, float]]     # 6 × (min, max) per joint


@dataclass
class MoveJPlan:
    """Result of a moveJ / PTP joint-interpolated motion plan.

    Attributes
    ----------
    q_start : ndarray (6,)
        Start joint configuration.
    q_end : ndarray (6,)
        End joint configuration (IK solution closest to q_start).
    path : ndarray (steps+1, 6)
        Joint samples from start to end, inclusive.
    violation_at : int or None
        Index of the first sample that violates joint limits,
        or None if all samples are within limits.
    joint_deltas : ndarray (6,)
        Per-joint absolute travel between start and end.
    """
    q_start: np.ndarray
    q_end: np.ndarray
    path: np.ndarray
    violation_at: Optional[int]
    joint_deltas: np.ndarray


def _to_joints6(joints: Joints6) -> Tuple[float, ...]:
    """Convert any sequence of 6 floats to a plain tuple for the Rust layer."""
    arr = np.asarray(joints, dtype=np.float64).ravel()
    if arr.shape != (6,):
        raise ValueError(f"Expected 6 joint values, got shape {arr.shape}")
    return tuple(float(v) for v in arr)


def _to_4x4_list(matrix: np.ndarray) -> list:
    """Convert (4,4) numpy array to nested Python list for the Rust layer."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"Expected (4,4) matrix, got {m.shape}")
    return m.tolist()


def _from_4x4_list(nested: list) -> Pose4x4:
    """Convert nested-list 4×4 result from Rust back to a numpy array."""
    return np.array(nested, dtype=np.float64)


def _interpolate_and_check(
    q_start: np.ndarray,
    q_end: np.ndarray,
    joint_limits: Optional[JointLimits],
    steps: int,
) -> MoveJPlan:
    """Linearly interpolate in joint space and check limits at every sample."""
    joint_deltas = np.abs(q_end - q_start)
    t = np.linspace(0.0, 1.0, steps + 1).reshape(-1, 1)
    path = q_start + t * (q_end - q_start)  # (steps+1, 6)

    violation_at: Optional[int] = None
    if joint_limits is not None:
        lo = np.array([lim[0] for lim in joint_limits], dtype=np.float64)
        hi = np.array([lim[1] for lim in joint_limits], dtype=np.float64)
        outside = (path < lo) | (path > hi)  # (steps+1, 6)
        bad_rows = np.where(outside.any(axis=1))[0]
        if len(bad_rows) > 0:
            violation_at = int(bad_rows[0])

    return MoveJPlan(
        q_start=q_start.copy(),
        q_end=q_end.copy(),
        path=path,
        violation_at=violation_at,
        joint_deltas=joint_deltas,
    )


class Robot:
    """
    OPW kinematics robot — numpy pose interface.

    Poses are represented as (4, 4) float64 NumPy arrays (homogeneous
    transformation matrices, row-major):

        [ R  t ]
        [ 0  1 ]

    Parameters
    ----------
    kinematic_model : KinematicModel
        Robot geometry built from DH-style OPW parameters.
    degrees : bool
        If True (default) joint angles are in degrees; otherwise radians.
    """

    def __init__(self, kinematic_model: KinematicModel, degrees: bool = True) -> None:
        self._robot = _RobotInternal(kinematic_model, degrees)
        self.degrees = degrees
        self.kinematic_model = kinematic_model

    def __repr__(self) -> str:
        return self._robot.__repr__()

    # ------------------------------------------------------------------
    # Single-pose methods
    # ------------------------------------------------------------------

    def forward(
        self,
        joints: Joints6,
        ee_transform: Optional[Pose4x4] = None,
    ) -> Pose4x4:
        """
        Forward kinematics for one joint configuration.

        Parameters
        ----------
        joints : array-like, shape (6,)
            Joint angles J1..J6 in degrees or radians.
        ee_transform : ndarray (4,4), optional
            End-effector offset transform appended after the TCP.

        Returns
        -------
        ndarray (4,4)
            Homogeneous transformation matrix of the TCP in world frame.
        """
        j6 = _to_joints6(joints)
        ee = None if ee_transform is None else _to_4x4_list(ee_transform)
        result = self._robot.forward(j6, ee)
        return _from_4x4_list(result)

    def inverse(
        self,
        pose: Pose4x4,
        current_joints: Optional[Joints6] = None,
        ee_transform: Optional[Pose4x4] = None,
    ) -> List[np.ndarray]:
        """
        Inverse kinematics for one target pose.

        Parameters
        ----------
        pose : ndarray (4,4)
            Desired TCP pose in world frame.
        current_joints : array-like (6,), optional
            Current joint angles for closest-solution ranking.
        ee_transform : ndarray (4,4), optional
            End-effector offset transform (same as used in forward).

        Returns
        -------
        list of ndarray (6,)
            All valid joint solutions. May be empty if unreachable.
        """
        pose_list = _to_4x4_list(pose)
        cj = None if current_joints is None else _to_joints6(current_joints)
        ee = None if ee_transform is None else _to_4x4_list(ee_transform)
        raw = self._robot.inverse(pose_list, cj, ee)
        return [np.array(sol, dtype=np.float64) for sol in raw]

    def forward_frames(
        self,
        joints: Joints6,
        ee_transform: Optional[Pose4x4] = None,
    ) -> List[Pose4x4]:
        """
        Forward kinematics for every link frame.

        Returns
        -------
        list of ndarray (4,4)
            Transforms for [Base, J1, J2, J3, J4, J5, J6, TCP] (8 frames).
        """
        j6 = _to_joints6(joints)
        ee = None if ee_transform is None else _to_4x4_list(ee_transform)
        raw = self._robot.forward_frames(j6, ee)
        return [_from_4x4_list(f) for f in raw]

    # ------------------------------------------------------------------
    # MoveJ / PTP planning
    # ------------------------------------------------------------------

    def move_j(
        self,
        start: Pose4x4,
        end: Pose4x4,
        previous: Joints6,
        joint_limits: Optional[JointLimits] = None,
        steps: int = 50,
        ee_transform: Optional[Pose4x4] = None,
    ) -> MoveJPlan:
        """
        Plan a joint-interpolated motion (moveJ / PTP) between two Cartesian poses.

        Solves IK at ``start`` seeded by ``previous`` to pick the start branch,
        then delegates to :meth:`move_j_from_joints`.

        Parameters
        ----------
        start : ndarray (4,4)
            Start TCP pose in world frame.
        end : ndarray (4,4)
            End TCP pose in world frame.
        previous : array-like (6,)
            Previous joint configuration used to seed the start-pose IK
            (preserves kinematic branch).
        joint_limits : sequence of 6 (min, max) tuples, optional
            Per-joint limits in the same unit as the robot (degrees or radians).
            When provided, every interpolated sample is checked.
        steps : int
            Number of interpolation intervals (``steps + 1`` samples).
        ee_transform : ndarray (4,4), optional
            End-effector offset transform.

        Returns
        -------
        MoveJPlan

        Raises
        ------
        ValueError
            If ``steps < 1`` or the start/end pose is unreachable.
        """
        if steps < 1:
            raise ValueError("steps must be >= 1")
        prev = _to_joints6(previous)
        ee = None if ee_transform is None else _to_4x4_list(ee_transform)
        solutions = self._robot.inverse(_to_4x4_list(start), prev, ee)
        if not solutions:
            raise ValueError("Start pose unreachable (no IK solution)")
        q_start = np.array(solutions[0], dtype=np.float64)
        return self.move_j_from_joints(q_start, end, joint_limits, steps, ee_transform)

    def move_j_from_joints(
        self,
        q_start: Joints6,
        end: Pose4x4,
        joint_limits: Optional[JointLimits] = None,
        steps: int = 50,
        ee_transform: Optional[Pose4x4] = None,
    ) -> MoveJPlan:
        """
        Plan a moveJ from a known start joint configuration to an end Cartesian pose.

        Useful when the robot's current joints are known (e.g. read from the
        controller) and you want to skip re-solving IK for the start pose.

        Parameters
        ----------
        q_start : array-like (6,)
            Start joint configuration.
        end : ndarray (4,4)
            End TCP pose in world frame.
        joint_limits : sequence of 6 (min, max) tuples, optional
            Per-joint limits in the same unit as the robot (degrees or radians).
        steps : int
            Number of interpolation intervals (``steps + 1`` samples).
        ee_transform : ndarray (4,4), optional
            End-effector offset transform.

        Returns
        -------
        MoveJPlan

        Raises
        ------
        ValueError
            If ``steps < 1`` or the end pose is unreachable.
        """
        if steps < 1:
            raise ValueError("steps must be >= 1")
        qs = np.asarray(q_start, dtype=np.float64).ravel()
        if qs.shape != (6,):
            raise ValueError(f"Expected 6 joint values, got shape {qs.shape}")
        cj = tuple(float(v) for v in qs)
        ee = None if ee_transform is None else _to_4x4_list(ee_transform)
        solutions = self._robot.inverse(_to_4x4_list(end), cj, ee)
        if not solutions:
            raise ValueError("End pose unreachable (no IK solution)")
        qe = np.array(solutions[0], dtype=np.float64)
        return _interpolate_and_check(qs, qe, joint_limits, steps)

    # ------------------------------------------------------------------
    # Batch methods
    # ------------------------------------------------------------------

    def batch_forward(
        self,
        joints: np.ndarray,
        ee_transform: Optional[Pose4x4] = None,
    ) -> np.ndarray:
        """
        Forward kinematics for N joint configurations.

        Parameters
        ----------
        joints : ndarray (N, 6)
            N rows of joint angles.
        ee_transform : ndarray (4,4), optional
            End-effector offset transform.

        Returns
        -------
        ndarray (N, 4, 4)
            One 4×4 TCP pose per row.
        """
        arr = np.ascontiguousarray(joints, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 6:
            raise ValueError(f"joints must be (N,6), got {arr.shape}")
        ee = None if ee_transform is None else _to_4x4_list(ee_transform)
        flat = self._robot.batch_forward(arr, ee)          # (N, 16)
        return flat.reshape(-1, 4, 4)

    def batch_inverse(
        self,
        poses: np.ndarray,
        current_joints: Optional[Joints6] = None,
        ee_transform: Optional[Pose4x4] = None,
    ) -> np.ndarray:
        """
        Inverse kinematics for N target poses (best solution per pose).

        Parameters
        ----------
        poses : ndarray (N, 4, 4) or (N, 16)
            N target TCP poses.
        current_joints : array-like (6,), optional
            Starting joint angles for solution continuity across the batch.
        ee_transform : ndarray (4,4), optional
            End-effector offset transform.

        Returns
        -------
        ndarray (N, 6)
            Best joint solution per pose. NaN rows where no solution exists.
        """
        arr = np.asarray(poses, dtype=np.float64)
        if arr.ndim == 3 and arr.shape[1:] == (4, 4):
            arr = arr.reshape(-1, 16)
        arr = np.ascontiguousarray(arr)
        if arr.ndim != 2 or arr.shape[1] != 16:
            raise ValueError(f"poses must be (N,4,4) or (N,16), got {arr.shape}")
        cj = None if current_joints is None else _to_joints6(current_joints)
        ee = None if ee_transform is None else _to_4x4_list(ee_transform)
        return self._robot.batch_inverse(arr, cj, ee)      # (N, 6)
