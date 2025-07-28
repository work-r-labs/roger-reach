import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import yaml
from typing import List


def matrix_from_rquat_t(r_quat, t) -> np.ndarray:
    """
    Convert a rotation quaternion and translation vector to a 4x4 transformation matrix. Quaternion is scalar last
    """
    r = R.from_quat(r_quat)
    r = r.as_matrix()
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = t
    return m


def matrix_to_rquat_t(m) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 4x4 transformation matrix to a rotation quaternion and translation vector. Quaternion is scalar last
    """
    r = R.from_matrix(m[:3, :3])
    r = r.as_quat(canonical=False)
    t = m[:3, 3]
    return r, t


def set_pose_orientation(hom_matrix: np.ndarray, orientation_quat: np.ndarray) -> np.ndarray:
    """
    Set the orientation of a homogeneous transformation matrix using a quaternion.
    Can be used to orient a transform to a frame, such as gripper down.

    Args:
        hom_matrix (numpy.ndarray): 4x4 homogeneous transformation matrix.
        orientation_quat (numpy.ndarray): Orientation quaternion. Scalar last

    Returns:
        numpy.ndarray: Updated 4x4 homogeneous transformation matrix.
    """
    hom_matrix = hom_matrix.copy()
    rotation_matrix = R.from_quat(orientation_quat).as_matrix()
    hom_matrix[:3, :3] = rotation_matrix
    return hom_matrix


def m_to_mm(m):
    """
    Convert meters to millimeters.
    """
    return m * 1000


def mm_to_m(mm):
    """
    Convert millimeters to meters.
    """
    return mm / 1000


def hm_m_to_mm(hm_m: np.ndarray) -> np.ndarray:
    """
    Convert a homogeneous transformation matrix from meters to millimeters.
    """
    hm_mm = hm_m.copy()
    hm_mm[:3, 3] *= 1000
    return hm_mm


def hm_mm_to_m(hm_mm: np.ndarray) -> np.ndarray:
    """
    Convert a homogeneous transformation matrix from millimeters to meters.
    """
    hm_m = hm_mm.copy()
    hm_m[:3, 3] /= 1000
    return hm_m


def tx(x: float | int) -> np.ndarray:
    """
    Create a homogeneous transformation matrix that translates by x in the x-axis.
    """
    return np.array(
        [
            [1, 0, 0, x],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def ty(y: float | int) -> np.ndarray:
    """
    Create a homogeneous transformation matrix that translates by y in the y-axis.
    """
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, y],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def tz(z: float | int) -> np.ndarray:
    """
    Create a homogeneous transformation matrix that translates by z in the z direction.
    """
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ]
    )


def txyz(x, y, z) -> np.ndarray:
    return tx(x) @ ty(y) @ tz(z)


def rx_deg(theta) -> np.ndarray:
    """
    Create a homogeneous transformation matrix for rotation around the x-axis.
    """
    r = R.from_euler("x", theta, degrees=True).as_matrix()
    m = np.eye(4)
    m[:3, :3] = r
    return m


def ry_deg(theta) -> np.ndarray:
    """
    Create a homogeneous transformation matrix for rotation around the y-axis.
    """
    r = R.from_euler("y", theta, degrees=True).as_matrix()
    m = np.eye(4)
    m[:3, :3] = r
    return m


def rz_deg(theta) -> np.ndarray:
    """
    Create a homogeneous transformation matrix for rotation around the z-axis.
    """
    r = R.from_euler("z", theta, degrees=True).as_matrix()
    m = np.eye(4)
    m[:3, :3] = r
    return m


def matrix_to_euler_xyz(matrix: np.ndarray) -> list[float]:
    """
    Given a 4x4 transformation matrix (as a list-of-lists or numpy array),
    compute the euler angles in degrees (XYZ order).

    params:
        matrix (list or np.ndarray): 4x4 homogeneous transformation matrix.

    returns:
        euler_angles (list): Euler angles in degrees (XYZ order).
    """

    # Extract the upper-left 3x3 rotation matrix
    rotation_matrix = matrix[:3, :3]

    # Convert to scipy Rotation object
    r = R.from_matrix(rotation_matrix)

    # Get Euler angles in degrees (XYZ order)
    euler_angles = r.as_euler("xyz", degrees=True)

    return euler_angles
