import numpy as np
import matplotlib.pyplot as plt


def compute_return_error(cumulants, predictions, gamma):
    num_time_steps = len(cumulants)
    returns = np.zeros(num_time_steps)
    returns[-1] = cumulants[-1]
    for t in range(num_time_steps - 2, -1, -1):
        returns[t] = gamma * returns[t + 1] + cumulants[t]
    return_error = (predictions - returns) ** 2
    MSRE = return_error.mean()
    return MSRE, return_error, returns


def plot_last_n(rewards_vec, predictions_vec, return_target, return_error, n=1000):
    fig, axs = plt.subplots(4, figsize=(30, 35))
    fig.tight_layout()
    axs[0].step(list(range(n)), rewards_vec[-n:])
    axs[1].step(list(range(n)), predictions_vec[-n:])
    axs[2].step(list(range(n)), return_target[-n:])
    axs[3].step(list(range(n)), return_error[-n:])

    axs[0].title.set_text("reward")
    axs[1].title.set_text("predictions")
    axs[2].title.set_text("true target")
    axs[3].title.set_text("error (prediction : true target)")
    for ax in axs:
        ax.grid(color="#666666", linestyle="-", alpha=0.5)
    return fig


def get_types(list_of_values):
    # returns the appropriate datatype needed by the cpp metric class
    list_of_types = []
    for value in list_of_values:
        if type(value) == str:
            list_of_types.append("VARCHAR(80)")
        elif type(value) == int:
            list_of_types.append("int")
        elif type(value) == float:
            list_of_types.append("real")
        else:
            raise NotImplementedError
    return list_of_types
