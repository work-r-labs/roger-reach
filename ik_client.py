import numpy as np
import requests


class IKClient:
    def __init__(self, server_url: str = "http://localhost:4567"):
        self.server_url = server_url

    def forward_kinematics(self, joints: np.ndarray) -> np.ndarray:
        """Compute forward kinematics for given joint angles."""
        response = requests.post(f"{self.server_url}/fk", json={"joints": joints.tolist()})
        response.raise_for_status()
        return np.array(response.json())

    def inverse_kinematics(self, pose_matrix: np.ndarray, previous_q: np.ndarray) -> np.ndarray:
        """Compute inverse kinematics for given pose matrix."""
        payload = {"matrix": pose_matrix.tolist(), "previous_q": previous_q.tolist()}
        response = requests.post(f"{self.server_url}/ik", json=payload)
        response.raise_for_status()
        return np.array(response.json())

    def batch_forward_kinematics(self, joints_batch: list) -> np.ndarray:
        """Compute forward kinematics for a batch of joint angles."""
        response = requests.post(f"{self.server_url}/fk_batch", json={"joints": joints_batch})
        response.raise_for_status()
        return np.array(response.json())

    def batch_inverse_kinematics(self, pose_matrices: list, previous_q: np.ndarray) -> np.ndarray:
        """Compute inverse kinematics for a batch of pose matrices."""
        pose_list = [pose.tolist() for pose in pose_matrices]
        payload = {"matrix": pose_list, "previous_q": previous_q.tolist()}
        response = requests.post(f"{self.server_url}/ik_batch", json=payload)
        response.raise_for_status()
        return np.array(response.json())
