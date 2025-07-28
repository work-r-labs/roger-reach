import numpy as np
import jax
import jaxlie
import jaxls
import pyroki as pk
from jax import lax
from jax import numpy as jnp


jax.config.update("jax_platform_name", "cpu")
jax.devices()
jnp.set_printoptions(suppress=True)

flange_to_gripper = jaxlie.SE3.from_translation(jnp.array([0.0, 0.0, -0.17]))


def newton_raphson(f, x, iters):
    """Use the Newton-Raphson method to find a root of the given function."""

    def update(x, _):
        y = x - f(x) / jax.grad(f)(x)
        return y, None

    x, _ = lax.scan(update, 1.0, length=iters)
    return x


def roberts_sequence(num_points, dim, root):
    # From https://gist.github.com/carlosgmartin/1fd4e60bed526ec8ae076137ded6ebab.
    basis = 1 - (1 / root ** (1 + jnp.arange(dim)))

    n = jnp.arange(num_points)
    x = n[:, None] * basis[None, :]
    x, _ = jnp.modf(x)

    return x


class PyrokiIkBeamHelper:
    def __init__(self, robot: pk.Robot, target_link_index: int):
        self.robot = robot
        exp = robot.joints.num_actuated_joints
        print(exp)
        self.root = newton_raphson(lambda x: x ** (exp + 1) - x - 1, 1.0, 10_000)
        self.target_link_index = jnp.array(target_link_index)

    # @jax.jit
    def solve_ik(self, target_wxyz: jax.Array, target_position: jax.Array, previous_q: jax.Array) -> jax.Array:
        num_seeds_init: int = 64
        num_seeds_final: int = 4

        total_steps: int = 16
        init_steps: int = 6

        def solve_one(
            initial_q: jax.Array, lambda_initial: float | jax.Array, max_iters: int
        ) -> tuple[jax.Array, jaxls.SolveSummary]:
            """Solve IK problem with a single initial condition. We'll vmap
            over initial_q to solve problems in parallel."""
            joint_var = robot.joint_var_cls(0)
            factors = [
                # pk.costs.pose_cost(
                pk.costs.pose_cost_analytic_jac(
                    robot,
                    joint_var,
                    jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(target_wxyz), target_position),
                    self.target_link_index,
                    pos_weight=10.0,
                    ori_weight=5.0,
                ),
                pk.costs.limit_cost(
                    robot,
                    joint_var,
                    weight=2.0,
                ),
                # pk.costs.manipulability_cost(robot, joint_var, jnp.array([6]), 0.1),
                pk.costs.rest_cost(
                    joint_var,
                    rest_pose=previous_q,
                    weight=jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.5]) * 0.1,
                    # rest_pose=jnp.deg2rad(jnp.array([0.0, 45.0, 0.00, 90.0, 0.0, 0.0])),
                    # weight=jnp.array(               [0.0, 1.00, 0.00, 1.0, 0.0, 0.0]) * 0.1
                ),
                # pk.costs.rest_cost(
                #     joint_var,
                #     rest_pose=jnp.deg2rad(jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
                #     weight=jnp.array(               [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]) * 0.1
                # rest_pose=jnp.deg2rad(jnp.array([0.0, 45.0, 0.00, 90.0, 0.0, 0.0])),
                # weight=jnp.array(               [0.0, 1.00, 0.00, 1.0, 0.0, 0.0]) * 0.1
                # )
            ]
            sol, summary = (
                jaxls.LeastSquaresProblem(factors, [joint_var])
                .analyze()
                .solve(
                    initial_vals=jaxls.VarValues.make([joint_var.with_value(initial_q)]),
                    verbose=False,
                    linear_solver="dense_cholesky",
                    termination=jaxls.TerminationConfig(
                        max_iterations=max_iters,
                        early_termination=False,
                    ),
                    trust_region=jaxls.TrustRegionConfig(lambda_initial=lambda_initial),
                    return_summary=True,
                )
            )
            return sol[joint_var], summary

        vmapped_solve = jax.vmap(solve_one, in_axes=(0, 0, None))

        # Create initial seeds, but this time with quasi-random sequence.
        robot = self.robot
        initial_qs = robot.joints.lower_limits + roberts_sequence(num_seeds_init, robot.joints.num_actuated_joints, self.root) * (
            robot.joints.upper_limits - robot.joints.lower_limits
        )

        # Optimize the initial seeds.
        initial_sols, summary = vmapped_solve(initial_qs, jnp.full(initial_qs.shape[:1], 10.0), init_steps)

        # Get the best initial solutions.
        best_initial_sols = jnp.argsort(summary.cost_history[jnp.arange(num_seeds_init), -1])[:num_seeds_final]

        # Optimize more for the best initial solutions.
        best_sols, summary = vmapped_solve(
            initial_sols[best_initial_sols],
            summary.lambda_history[jnp.arange(num_seeds_init), -1][best_initial_sols],
            total_steps - init_steps,
        )
        return best_sols[jnp.argmin(summary.cost_history[jnp.arange(num_seeds_final), summary.iterations])]

    def forward_kinematics(self, q: jax.Array | np.ndarray) -> jax.Array:
        return self.robot.forward_kinematics(jnp.asarray(q))[self.target_link_index]
