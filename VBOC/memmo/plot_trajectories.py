import numpy as np
import matplotlib.pyplot as plt


def plot_trajectories(trajectories):
    fig, ax = plt.subplots(3, 1)
    for traj in trajectories:
        for i in range(3):
            ax[i].plot(traj[:, i], traj[:, i+3], color='blue', alpha=0.3)
            ax[i].scatter(traj[0, i], traj[0, i + 3], color='red', marker='x')
            ax[i].scatter(traj[-1, i], traj[-1, i+3], color='green', marker='o')
    ax[0].set_xlabel('q1')
    ax[0].set_ylabel('q1_dot')
    ax[1].set_xlabel('q2')
    ax[1].set_ylabel('q2_dot')
    ax[2].set_xlabel('q3')
    ax[2].set_ylabel('q3_dot')
    plt.show()


if __name__ == '__main__':
    x_traj = np.load('../traj_3dof_vboc.npy')
    plot_trajectories(x_traj[:100])
