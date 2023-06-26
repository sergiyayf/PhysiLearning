from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def plot_tensorboard_logs(file_path='./', scalar_tags=['rollout/ep_len_mean'], **kwargs):
    """Plots the scalars from a tensorboard log file.

    Parameters
    ----------
    file_path: str
        Path to the tensorboard file.
    scalar_tags: list or array-like of str
        List or array-like of scalar tags to plot.

    Returns
    -------
    fig: matplotlib.figure.Figure
        Figure object containing the plot.

    """

    # Load the tensorboard file
    event_acc = EventAccumulator(file_path)
    event_acc.Reload()
    summary_iterators = [event_acc.Scalars(tag) for tag in scalar_tags]

    # Set up the plot
    fig, axs = plt.subplots(len(scalar_tags), 1, figsize=(10, len(scalar_tags) * 5))

    # Loop through the tags and plot the data
    for i, tag in enumerate(scalar_tags):
        steps = [event.step for event in summary_iterators[i]]
        values = [event.value for event in summary_iterators[i]]
        axs[i].plot(steps, values, **kwargs)
        axs[i].set_title(tag)

    return fig

file_path = '/home/saif/Projects/PhysiLearning/data/raven_LSTM_try/rPPO_LSTM_dont_give_treatment_1'
scalar_tags = ['rollout/ep_len_mean', 'rollout/ep_rew_mean']
plot_tensorboard_logs(file_path, scalar_tags, color='red')