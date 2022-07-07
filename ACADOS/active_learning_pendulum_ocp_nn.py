import cProfile
import numpy as np
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
import time
from pendulum_ocp_class import OCPpendulumINIT
import warnings
import random
import queue
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from my_nn import NeuralNet
from my_nn import NNTrain

# import queue
# import time

warnings.filterwarnings("ignore")

with cProfile.Profile() as pr:

    start_time = time.time()

    # Ocp initialization:
    ocp = OCPpendulumINIT()

    ocp_dim = ocp.nx  # state space dimension

    # Position and velocity bounds:
    v_max = ocp.dthetamax
    v_min = -ocp.dthetamax
    q_max = ocp.thetamax
    q_min = ocp.thetamin

    # Hyper-parameters for nn:
    input_size = ocp_dim
    hidden_size = ocp_dim * 20
    output_size = 2
    learning_rate = 0.01

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Active learning parameters:
    N_init = pow(10, ocp_dim)  # size of initial labeled set
    B = pow(10, ocp_dim)  # batch size
    etp_stop = 0.1  # active learning stopping condition
    loss_stop = 0.08  # nn training stopping condition
    beta = 0.8
    n_minibatch = 32
    n_epoch = 100

    # Generate low-discrepancy unlabeled samples:
    sampler = qmc.Halton(d=ocp_dim, scramble=False)
    sample = sampler.random(n=pow(100, ocp_dim))
    l_bounds = [q_min, v_min]
    u_bounds = [q_max, v_max]
    data = qmc.scale(sample, l_bounds, u_bounds)

    Xu_iter = data  # Unlabeled set
    Xu_iter_tensor = torch.from_numpy(Xu_iter.astype(np.float32))
    mean, std = torch.mean(Xu_iter_tensor), torch.std(Xu_iter_tensor)

    # Generate the initial set of labeled samples:
    X_iter = np.empty((10 * 2 * ocp_dim, ocp_dim))
    y_iter = np.full((10 * 2 * ocp_dim, 2), [1, 0])
    q_test = np.linspace(q_min, q_max, num=10)
    v_test = np.linspace(v_min, v_max, num=10)

    for p in range(10):
        X_iter[p, :] = [q_min - 0.1, v_test[p]]
        X_iter[p + 10, :] = [q_max + 0.1, v_test[p]]
        X_iter[p + 20, :] = [q_test[p], v_min - 1]
        X_iter[p + 30, :] = [q_test[p], v_max + 1]

    # # Generate the initial set of labeled samples:
    # res = ocp.compute_problem((q_max + q_min) / 2, 0.0)
    # if res != 2:
    #     X_iter = np.double([[(q_max + q_min) / 2, 0.0]])
    #     if res == 1:
    #         y_iter = [[0, 1]]
    #     else:
    #         y_iter = [[1, 0]]
    # else:
    #     raise Exception("Max iteration reached")

    # Training of an initial classifier:
    for n in range(N_init):
        q0 = data[n, 0]
        v0 = data[n, 1]

        # Data testing:
        res = ocp.compute_problem(q0, v0)
        if res != 2:
            X_iter = np.append(X_iter, [[q0, v0]], axis=0)
            if res == 1:
                y_iter = np.append(y_iter, [[0, 1]], axis=0)
            else:
                y_iter = np.append(y_iter, [[1, 0]], axis=0)
        else:
            raise Exception("Max iteration reached")

        # Add intermediate states of succesfull initial conditions
        if res == 1:
            for f in range(1, ocp.N, int(ocp.N / 3)):
                current_val = ocp.ocp_solver.get(f, "x")
                X_iter = np.append(X_iter, [current_val], axis=0)
                y_iter = np.append(y_iter, [[0, 1]], axis=0)

    # Delete tested data from the unlabeled set:
    Xu_iter = np.delete(Xu_iter, range(N_init), axis=0)

    val = 1

    X_iter_tensor = torch.from_numpy(X_iter.astype(np.float32))
    y_iter_tensor = torch.from_numpy(y_iter.astype(np.float32))
    X_iter_tensor = (X_iter_tensor - mean) / std

    my_dataset = TensorDataset(X_iter_tensor, y_iter_tensor)
    my_dataloader = DataLoader(my_dataset, batch_size=n_minibatch, shuffle=True)

    NNTrain(my_dataloader, optimizer, model, criterion, n_epoch, val, beta, loss_stop)

    # X_iter_tensor = torch.from_numpy(X_iter.astype(np.float32))
    # y_iter_tensor = torch.from_numpy(y_iter.astype(np.float32))
    # X_iter_tensor = (X_iter_tensor - mean) / std

    # q = queue.Queue()

    # it = 0

    # # Train the model
    # while val > loss_stop and it <= 1000:
    #     it += 1

    #     # Forward pass
    #     outputs = model(X_iter_tensor)
    #     loss = criterion(outputs, y_iter_tensor)

    #     # Backward and optimize
    #     for param in model.parameters():
    #         param.grad = None

    #     loss.backward()
    #     optimizer.step()

    #     val = loss.item()

    #     if q.qsize() >= 100:
    #         st = q.get()
    #         if round(st, 3) == round(val, 3):
    #             print("Iteration skipped")
    #             continue

    # it = 0

    # # Train the model
    # while val > loss_stop and it <= 1000:

    #     ind = random.sample(range(X_iter.shape[0]), n_minibatch)
    #     X_iter_tensor = torch.from_numpy(X_iter[ind].astype(np.float32))
    #     y_iter_tensor = torch.from_numpy(y_iter[ind].astype(np.float32))
    #     X_iter_tensor = (X_iter_tensor - mean) / std

    #     # Forward pass
    #     outputs = model(X_iter_tensor)
    #     loss = criterion(outputs, y_iter_tensor)

    #     # Backward and optimize
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()

    #     val = beta * val + (1 - beta) * loss.item()

    #     it += 1

    print("INITIAL CLASSIFIER TRAINED")

    with torch.no_grad():
        # Plot the results:
        plt.figure()
        x_min, x_max = 0.0, np.pi / 2
        y_min, y_max = -10.0, 10.0
        h = 0.02
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        inp = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
        inp = (inp - mean) / std
        out = model(inp)
        y_pred = np.argmax(out.numpy(), axis=1)
        Z = y_pred.reshape(xx.shape)
        z = [
            0 if np.array_equal(y_iter[x], [1, 0]) else 1
            for x in range(y_iter.shape[0])
        ]
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        scatter = plt.scatter(
            X_iter[:, 0], X_iter[:, 1], c=z, marker=".", alpha=0.5, cmap=plt.cm.Paired
        )
        plt.xlim([0.0, np.pi / 2 - 0.01])
        plt.ylim([-10.0, 10.0])
        plt.xlabel("Initial position [rad]")
        plt.ylabel("Initial velocity [rad/s]")
        plt.title("Classifier")
        hand = scatter.legend_elements()[0]
        plt.legend(handles=hand, labels=("Non viable", "Viable"))

        # # Plot of the entropy:
        # plt.figure()
        # sigmoid = nn.Sigmoid()
        # prob_xu = sigmoid(out).numpy()
        # etp_xu = entropy(prob_xu, axis=1)
        # out = etp_xu.reshape(xx.shape)
        # levels = np.linspace(out.min(), out.max(), 10)
        # plt.contourf(xx, yy, out, levels=levels)
        # this = plt.contour(xx, yy, out, levels=levels, colors=("k",), linewidths=(1,))
        # plt.clabel(this, fmt="%2.1f", colors="w", fontsize=11)
        # plt.xlim([0.0, np.pi / 2 - 0.01])
        # plt.ylim([-10.0, 10.0])
        # plt.xlabel("Initial position [rad]")
        # plt.ylabel("Initial velocity [rad/s]")
        # plt.title("Entropy")

    # Active learning:
    k = 0  # iteration number
    etpmax = 1
    performance_history = [etpmax]

    while not (etpmax < etp_stop or Xu_iter.shape[0] == 0):

        if Xu_iter.shape[0] < B:
            B = Xu_iter.shape[0]

        with torch.no_grad():
            sigmoid = nn.Sigmoid()
            Xu_iter_tensor = torch.from_numpy(Xu_iter.astype(np.float32))
            Xu_iter_tensor = (Xu_iter_tensor - mean) / std
            prob_xu = sigmoid(model(Xu_iter_tensor)).numpy()
            etp = entropy(prob_xu, axis=1)

        maxindex = np.argpartition(etp, -B)[-B:]  # indexes of the uncertain samples

        etpmax = max(etp)  # max entropy used for the stopping condition
        performance_history.append(etpmax)

        k += 1

        # Add the B most uncertain samples to the labeled set:
        for x in range(B):
            q0 = Xu_iter[maxindex[x], 0]
            v0 = Xu_iter[maxindex[x], 1]

            # Data testing:
            res = ocp.compute_problem(q0, v0)
            if res != 2:
                X_iter = np.append(X_iter, [[q0, v0]], axis=0)
                if res == 1:
                    y_iter = np.append(y_iter, [[0, 1]], axis=0)
                else:
                    y_iter = np.append(y_iter, [[1, 0]], axis=0)
            else:
                raise Exception("Max iteration reached")

        # Delete tested data from the unlabeled set:
        Xu_iter = np.delete(Xu_iter, maxindex, axis=0)

        val = 1

        selec = [i for i in range(X_iter.shape[0] - B) if np.random.uniform() <= 1 / k]
        selec.extend([i for i in range(X_iter.shape[0] - B, X_iter.shape[0])])

        X_iter_tensor = torch.from_numpy(X_iter[selec].astype(np.float32))
        X_iter_tensor = (X_iter_tensor - mean) / std
        y_iter_tensor = torch.from_numpy(y_iter[selec].astype(np.float32))

        my_dataset = TensorDataset(X_iter_tensor, y_iter_tensor)
        my_dataloader = DataLoader(my_dataset, batch_size=n_minibatch, shuffle=True)

        NNTrain(
            my_dataloader, optimizer, model, criterion, n_epoch, val, beta, loss_stop
        )

        # selec = [i for i in range(X_iter.shape[0] - B) if np.random.uniform() <= 1 / k]
        # selec.extend([i for i in range(X_iter.shape[0] - B, X_iter.shape[0])])

        # X_iter_tensor = torch.from_numpy(X_iter[selec].astype(np.float32))
        # X_iter_tensor = (X_iter_tensor - mean) / std
        # y_iter_tensor = torch.from_numpy(y_iter[selec].astype(np.float32))

        # q = queue.Queue()

        # it = 0

        # # Train the model
        # while val > loss_stop and it <= 1000:
        #     it += 1

        #     # Forward pass
        #     outputs = model(X_iter_tensor)
        #     loss = criterion(outputs, y_iter_tensor)

        #     # Backward and optimize
        #     for param in model.parameters():
        #         param.grad = None

        #     loss.backward()
        #     optimizer.step()

        #     val = loss.item()

        #     q.put(val)

        #     if q.qsize() >= 100:
        #         st = q.get()
        #         if round(st, 3) == round(val, 3):
        #             print("Iteration skipped")
        #             continue

        # it = 0

        # # Train the model
        # while val > loss_stop and it <= 1000:

        #     ind = random.sample(range(X_iter.shape[0] - B), int(n_minibatch / 2))
        #     ind.extend(
        #         random.sample(
        #             range(X_iter.shape[0] - B, X_iter.shape[0]), int(n_minibatch / 2)
        #         )
        #     )
        #     X_iter_tensor = torch.from_numpy(X_iter[ind].astype(np.float32))
        #     y_iter_tensor = torch.from_numpy(y_iter[ind].astype(np.float32))
        #     X_iter_tensor = (X_iter_tensor - mean) / std

        #     # Forward pass
        #     outputs = model(X_iter_tensor)
        #     loss = criterion(outputs, y_iter_tensor)

        #     # Backward and optimize
        #     loss.backward()
        #     optimizer.step()
        #     optimizer.zero_grad()

        #     val = beta * val + (1 - beta) * loss.item()

        #     it += 1

        print("CLASSIFIER", k, "TRAINED")

    with torch.no_grad():
        # Plot the results:
        plt.figure()
        out = model(inp)
        y_pred = np.argmax(out.numpy(), axis=1)
        Z = y_pred.reshape(xx.shape)
        z = [
            0 if np.array_equal(y_iter[x], [1, 0]) else 1
            for x in range(y_iter.shape[0])
        ]
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        scatter = plt.scatter(
            X_iter[:, 0],
            X_iter[:, 1],
            c=z,
            marker=".",
            alpha=0.5,
            cmap=plt.cm.Paired,
        )
        plt.xlim([0.0, np.pi / 2 - 0.01])
        plt.ylim([-10.0, 10.0])
        plt.xlabel("Initial position [rad]")
        plt.ylabel("Initial velocity [rad/s]")
        plt.title("Classifier")
        hand = scatter.legend_elements()[0]
        plt.legend(handles=hand, labels=("Non viable", "Viable"))

        # # Plot of the entropy:
        # plt.figure()
        # prob_xu = sigmoid(out).numpy()
        # etxu = entropy(prob_xu, axis=1)
        # out = etxu.reshape(xx.shape)
        # levels = np.linspace(out.min(), out.max(), 10)
        # plt.contourf(xx, yy, out, levels=levels)
        # this = plt.contour(xx, yy, out, levels=levels, colors=("k",), linewidths=(1,))
        # plt.clabel(this, fmt="%2.1f", colors="w", fontsize=11)
        # plt.xlim([0.0, np.pi / 2 - 0.01])
        # plt.ylim([-10.0, 10.0])
        # plt.xlabel("Initial position [rad]")
        # plt.ylabel("Initial velocity [rad/s]")
        # plt.title("Entropy")

    plt.figure()
    plt.plot(performance_history[1:])
    plt.scatter(range(len(performance_history[1:])), performance_history[1:])
    plt.xlabel("Iteration number")
    plt.ylabel("Maximum entropy")
    plt.title("Maximum entropy evolution")

    print("Execution time: %s seconds" % (time.time() - start_time))

# pr.print_stats(sort="cumtime")

plt.show()