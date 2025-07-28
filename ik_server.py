from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import numpy as np
import jax
import jax.numpy as jnp
import jaxlie
from jaxlie import SE3
import pyroki as pk
import yourdfpy
from ik_beam_helper import PyrokiIkBeamHelper
from typing import List

jax.config.update("jax_platform_name", "cpu")
jax.devices()
jnp.set_printoptions(suppress=True)


class PyRoKiPlanner:
    def __init__(self, urdf_path: Path, target_link_name: str):
        self.urdf_path = urdf_path
        assert urdf_path.exists(), f"{urdf_path=} does not exist"
        self.urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=urdf_path.parent / "meshes")
        self.robot = pk.Robot.from_urdf(self.urdf)

        self.target_link_name = target_link_name
        self.target_link_index = self.robot.links.names.index(self.target_link_name)

        self.robot_coll = pk.collision.RobotCollision.from_urdf(self.urdf)

        self.ik_helper = PyrokiIkBeamHelper(self.robot, self.target_link_index)
        self._single_ik = jax.jit(self.ik_helper.solve_ik)
        self._batched_ik = jax.jit(jax.vmap(self.ik_helper.solve_ik))
        self._single_fk = jax.jit(self.ik_helper.forward_kinematics)
        self._batched_fk = jax.jit(jax.vmap(self.ik_helper.forward_kinematics))

    @property
    def ndof(self) -> int:
        return self.urdf.num_actuated_joints

    def fk(self, q: jax.Array) -> SE3:
        return SE3(self._single_fk(q))

    def batched_fk(self, q: jax.Array) -> SE3:
        return SE3(self._batched_fk(q))

    def ik(self, pose: SE3, previous_q: jax.Array | None = None) -> jax.Array:
        target_wxyz, target_position = pose.wxyz_xyz[:4], pose.translation()
        if previous_q is None:
            previous_q = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return self._single_ik(target_wxyz, target_position, previous_q)

    def batched_ik(self, poses: SE3, previous_q: jax.Array | None = None) -> jax.Array:
        target_wxyz, target_position = poses.wxyz_xyz[:, :4], poses.translation()
        if previous_q is None:
            previous_q = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return self._batched_ik(target_wxyz, target_position, previous_q)


robot_library = Path.home() / "robots"

robot_urdf_path = robot_library / "library/ABB/CRB15000_10kg_152_v1/CRB15000_10kg_152.urdf" 
assert robot_urdf_path.exists(), f"{robot_urdf_path} not exist"
planner = PyRoKiPlanner(robot_urdf_path, "flange")


class JointAnglesRequest(BaseModel):
    joints: List[float]


class BatchJointAnglesRequest(BaseModel):
    joints: List[List[float]]


class PoseRequest(BaseModel):
    matrix: List[List[float]]
    previous_q: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class BatchPoseRequest(BaseModel):
    matrix: List[List[List[float]]]
    previous_q: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


port = 4567
app = FastAPI()


@app.get("/")
def home():
    return {"message": "IK/FK Server"}


@app.post("/ik")
def ik_route(request: PoseRequest):
    pose_matrix = jnp.array(request.matrix)
    pose = SE3.from_matrix(pose_matrix)
    previous_q = jnp.array(request.previous_q)
    ret = planner.ik(pose, previous_q)
    return ret.tolist()


@app.post("/ik_batch")
def ik_batch_route(request: BatchPoseRequest):
    pose_matrices = jnp.array(request.matrix)
    poses = SE3.from_matrix(pose_matrices)
    previous_q = jnp.array(request.previous_q)
    ret = planner.batched_ik(poses, previous_q)
    return ret.tolist()


@app.post("/fk_batch")
def fk_batch_route(request: BatchJointAnglesRequest):
    joints_batch = jnp.array(request.joints)
    ee_batch = planner.batched_fk(joints_batch)
    return ee_batch.as_matrix().tolist()


@app.post("/fk")
def fk_route(request: JointAnglesRequest):
    joints = jnp.array(request.joints)
    ee = planner.fk(joints)
    return ee.as_matrix().tolist()


if __name__ == "__main__":
    uvicorn.run("ik_server:app", port=port)
