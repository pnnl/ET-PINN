import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def historyPlot(title, filepath, linstyles='-*', num=None, save=None):
    """
    Plots historical data (e.g., training loss, validation metrics) from a CSV file.
    The CSV file is expected to have a header row. The first column(s) often correspond
    to some sort of index (e.g., step, time, epoch), followed by one or more columns
    of values (e.g., loss, accuracy, etc.).

    Args:
        title (str): Title for the plot.
        filepath (str): Path to the CSV file containing the data.
        linstyles (str, optional): Line style string, passed to matplotlib plot function. Default is '-*'.
        num (int, optional): Figure number. If provided, the figure with this number is cleared and reused.
        save (str, optional): File path to save the figure. If not None, saves the figure after plotting.

    Returns:
        num (int): The figure number used for the plot.
    """
    # Read data from CSV into a DataFrame. 
    df = pd.read_csv(filepath, header=0, sep="\s*,\s*", engine='python')
    
    # The first column is often time/epoch, the second is often step, and subsequent columns are values.
    step = df.iloc[:, 1]
    
    # values are the metrics we want to plot. df.iloc[:,2:] takes all columns from the third onward.
    values = df.iloc[:, 2:]
    
    # If a figure number is provided, clear that figure; otherwise, create a new figure.
    if num is not None:
        plt.figure(num=num, clear=True)
    else:
        num = plt.figure().number

    # Extract the names of the metrics from the DataFrame columns
    names = df.columns.values[2:]
    nname = len(names)

    # Plot values against step. Using semilogy for a log-scale y-axis.
    # If multiple metrics, it plots them all on the same plot.
    plt.semilogy(np.asarray(step), np.asarray(values), linstyles, label=names[0] if nname == 1 else names)
    plt.legend(loc='best')
    plt.title(title)
    plt.show()

    # If a save path is provided, save the figure.
    if save is not None:
        plt.savefig(save, bbox_inches='tight', pad_inches=0.1)
    return num


def SAplot(title, X, SAweights, save=None):
    """
    Plots scatter plots of SAweights (e.g., adaptive weights)
    against the input domain X.

    If SAweights has multiple columns (e.g., multiple weight sets), it will produce multiple plots.

    Args:
        title (str): Base title for the plots.
        X (ndarray): Numpy array of input coordinates of shape (N, d), where N is number of points and d is dimension.
        SAweights (ndarray): Numpy array of shape (N, M), where M is number of weight sets to plot.
        save (str, optional): File path to save the figure(s). If multiple sets are provided (M>1), 
                              it will append the set index to the filename.

    Returns:
        None
    """
    n = SAweights.shape[1]
    if save is not None:
        # Split the save path into name and extension
        tmp = save.split(".")
        for i in range(n):
            ititle = title + "-" + str(i) if n > 1 else title
            # If multiple weight sets, modify filename by appending the index.
            isave = ".".join(tmp[:-1]) + str(i) + "." + tmp[-1] if n > 1 else save
            # scatterPlot creates and shows the figure. We then save and close it.
            plot = scatterPlot(ititle, [X], [SAweights[:, i:i+1]])
            if plot:
                plt.savefig(isave, bbox_inches='tight', pad_inches=0.1)
                plt.close(plot)
    return


def scatterPlot(title, xyarr, carr, s=10, cm='rainbow', marker='^'):
    """
    Creates a scatter plot of points with colors determined by associated scalar values.

    This function can handle 1D or 2D inputs:
    - If dimension is 1D, points are plotted on a line (y=x if dimension=1).
    - If dimension is 2D, points are scattered in the 2D plane.

    Args:
        title (str): Title for the plot.
        xyarr (list of np.ndarray): A list of arrays containing coordinates. 
                                    Each array should have shape (N, d) with d=1 or 2.
        carr (list of np.ndarray): A list of arrays with shape (N, 1) that give colors/values for each point.
        s (int, optional): Marker size.
        cm (str, optional): Colormap name.
        marker (str, optional): Marker style, default is '^'.

    Returns:
        fig (matplotlib.figure.Figure or bool): The figure instance if the plot was successful, 
                                                or False if dimension > 2 and plot couldn't be made.
    """
    # Determine the min and max across all color arrays to set colorbar limits.
    vmin = min(c.min() for c in carr)
    vmax = max(c.max() for c in carr)
    dim = xyarr[0].shape[1]

    if dim > 2:
        print("scatterPlot can only be applied to 1D and 2D")
        return False

    fig, ax = plt.subplots()
    for xy, c in zip(xyarr, carr):
        if dim == 1:
            # For 1D, we plot y=x to visualize values along a line.
            im = ax.scatter(xy[:, 0], xy[:, 0], s=s, c=c, vmin=vmin, vmax=vmax, cmap=cm, marker=marker)
        elif dim == 2:
            # For 2D, scatter points in the xy plane.
            im = ax.scatter(xy[:, 0], xy[:, 1], s=s, c=c, vmin=vmin, vmax=vmax, cmap=cm, marker=marker)

    # Add a colorbar for the values.
    fig.colorbar(im, ax=ax)
    plt.title(title)
    plt.show()
    return fig
