from dataclasses import dataclass
import numpy as np
from pathlib import Path
from isaacsim import SimulationApp
from typing import Generator

simuluation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, XFormPrim
from isaacsim.core.utils.types import ArticulationAction
from ik_client import IKClient
from scipy.spatial.transform import Rotation as R
import spatial_utils as su

np.set_printoptions(suppress=True)

project_root = Path(__file__).parent
robot_library = project_root / "robots/library"


def matrix_from_xform(xform: XFormPrim) -> np.ndarray:
    xyz, quat = xform.get_world_poses()
    wxyz = quat[0]
    xyzw = [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]
    rmat = R.from_quat(xyzw).as_matrix()
    m = np.eye(4)
    m[:3, 3] = xyz
    m[:3, :3] = rmat
    return m


def matrix_from_prim_path(prim_path: str) -> np.ndarray:
    xform = XFormPrim(prim_path)
    return matrix_from_xform(xform)


def articulation_from_prim_path(prim_path: str) -> SingleArticulation:
    add_reference_to_stage(str(robot_usd_path), prim_path)
    return SingleArticulation(prim_path)

world = World()

world.scene.add_default_ground_plane() # type: ignore

# setup stage
scene_usd_path = project_root / "scene.usd"
assert scene_usd_path.exists()
scene_prim_path = "/World"
add_reference_to_stage(str(scene_usd_path), scene_prim_path)

# robot setup
robot_usd_path = robot_library / "ABB/CRB15000_10kg_152_v1/CRB15000_10kg_152/CRB15000_10kg_152.usd"
assert robot_usd_path.exists()
robot_mount_prim_path = f"{scene_prim_path}/robot_mount"
robot_prim_path = f"{robot_mount_prim_path}/robot"
add_reference_to_stage(str(robot_usd_path), robot_prim_path)

robot = SingleArticulation(robot_prim_path)

target_prim_paths: list[str] = [
    f"{scene_prim_path}/stand0/target",
    f"{scene_prim_path}/tote2/target",
    f"{scene_prim_path}/tote1/target",
    f"{scene_prim_path}/conv/target"
]

@dataclass
class Action:
    q: np.ndarray | None  # radians
    dt: float  # seconds

    def with_dt(self, dt: float):
        return Action(q=self.q, dt=dt)

    def get_q(self) -> np.ndarray:
        assert self.q is not None
        return self.q

    @property
    def art(self) -> ArticulationAction | None:
        if self.q is not None:
            return ArticulationAction(self.q)
        else:
            return None

ik_client = IKClient()

def action_generator() -> Generator[Action, None, None]:
    for target_prim_path in target_prim_paths:
        world_to_robot = matrix_from_prim_path(robot_mount_prim_path)
        world_to_target = matrix_from_prim_path(target_prim_path)

        # calculate robot to target
        robot_to_target = np.linalg.inv(world_to_robot) @ world_to_target
        robot_to_target_approach = robot_to_target @ su.tz(-0.2)

        q_approach = ik_client.inverse_kinematics(robot_to_target_approach, np.zeros(6))
        q = ik_client.inverse_kinematics(robot_to_target, np.zeros(6))

        yield Action(q_approach, 2)
        yield Action(q, 1)
        yield Action(q_approach, 1)


world.reset()
robot.initialize()
i = 0
while True:
    for action in action_generator():
        if action.art:
            robot.apply_action(action.art)
        for _ in range(int(action.dt * 60)):
            world.step(render=True)
            i += 1
    
