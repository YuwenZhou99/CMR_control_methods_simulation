import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from ur_simulation.classic_control.robots.ur7e import UR7e


def create_scene():
    init_joint_angles = np.array([1.57, -1.7, 2.4, -1.57, -1.57, -1.57])
    robot = UR7e(
        block_gripper=True,
        neutral_joints=init_joint_angles,
    )
    robot.sim.create_plane(0)
    return robot


def inverse_dynamics(q, dq, q_des, dynamics, kp, kd):
    M = dynamics.mass_matrix
    C = dynamics.coriolis_vector
    g = dynamics.gravity_vector

    # Calculate the expected acceleration
    y = kp * (q_des - q) + kd * (-dq)

    # Convert desired acceleration into joint torques
    tau = (M @ y) + C + g
    return tau


def stiffness_control():
    # TODO: implement stiffness control here.
    pass


def run(controller, target):
    if controller == "stiffness":
        stiffness_control()
        print("Stiffness control is not implemented yet.")
        return

    robot = create_scene()

    # Desired joint angles
    q_des = np.array(target)

    # Diagonal gain matrices for PD control
    kp = np.array([100, 100, 100, 100, 100, 100])
    kd = np.array([10, 10, 10, 10, 10, 10])

    duration = 5.0  # Simulation duration in seconds.
    times, joint_angles = [], []
    joint_velocities, commanded_torques, applied_torques = [], [], []
    step = 0
    try:
        while True:
            if step * robot.sim.dt >= duration:
                break
            # Get the current joint angles and velocities
            q = robot.robot_state.get_joint_angles()
            dq = robot.robot_state.get_joint_velocities()

            # Get the current dynamics model
            dynamics = robot.robot_model.get_dynamics()
            tau = inverse_dynamics(q, dq, q_des, dynamics, kp, kd)

            times.append(step * robot.sim.dt)
            joint_angles.append(q)
            joint_velocities.append(dq)
            commanded_torques.append(tau)
            applied_torques.append(np.clip(tau, -robot.arm_joint_forces, robot.arm_joint_forces))
            # Apply the torques and step the simulation
            robot.control_torques(tau)
            robot.sim.step()
            step += 1
    finally:
        robot.sim.close()  # Close the animation before showing the plots.


    joint_angles = np.array(joint_angles)
    desired_angles = np.tile(q_des, (len(times), 1))
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for i, ax in enumerate(axes.flat):
        # q_i: desired angle; joint_angles_i: measured angle.
        ax.plot(times, desired_angles[:, i], color="#66c2a5", linewidth=1, label=f"q_{i} (desired)")
        ax.plot(times, joint_angles[:, i], color="coral", linewidth=1, label=f"joint_angles_{i} (actual)")
        ax.legend()
        ax.set_xlabel("t")
        ax.set_ylabel(f"q_{i}, joint_angles_{i}")
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("0.6")
        ax.tick_params(colors="0.5", length=0)
    fig.tight_layout()
    results = Path(__file__).resolve().parents[1] / "results"
    results.mkdir(exist_ok=True)
    fig.savefig(results / f"{controller}.png", dpi=150)
    # All six joints; tau_applied is the clipped command, not a measured torque.
    np.savez_compressed(results / f"{controller}.npz",
                        time=np.array(times), q=joint_angles, q_des=desired_angles,
                        dq=np.array(joint_velocities), error=desired_angles - joint_angles,
                        tau_command=np.array(commanded_torques),
                        tau_applied=np.array(applied_torques))
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=["inverse_dynamics", "stiffness"],
                        default="inverse_dynamics")
    parser.add_argument("--target", nargs=6, type=float,
                        default=[1.77, -1.7, 2.4, -1.57, -1.57, -1.57])
    args = parser.parse_args()
    run(args.controller, args.target)
