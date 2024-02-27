import constants

import pandas as pd
import matplotlib.dates
import matplotlib.pyplot as plt


def incidents_over_years(dataframe: pd.DataFrame, figsize: tuple[float, float] = [12, 6], limit_left: str = None, limit_right: str = None):
    # NB! temporary method for OUS meeting
    # prepare data
    daily_incidents = dataframe.groupby(dataframe["time_call_received"].dt.date).size()

    # plot data
    matplotlib.pyplot.figure(figsize=figsize)
    matplotlib.pyplot.plot(daily_incidents.index, daily_incidents.values)

    # aesthetics
    if limit_left is not None:
        matplotlib.pyplot.xlim(left=pd.Timestamp(limit_left))
    if limit_right is not None:
        matplotlib.pyplot.xlim(right=pd.Timestamp(limit_right))

    matplotlib.pyplot.gca().xaxis.set_major_locator(matplotlib.dates.YearLocator())
    matplotlib.pyplot.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))

    matplotlib.pyplot.grid(True)

    # set labels
    matplotlib.pyplot.title("Total Incidents Per Day Over Years", fontdict=constants.FONT_PROPERTIES_HEADER)

    matplotlib.pyplot.xlabel("Year")
    matplotlib.pyplot.ylabel("Total Incidents")

    # save/show plot
    matplotlib.pyplot.show()


def overlay_incidents_over_years(dataframe1: pd.DataFrame, dataframe2: pd.DataFrame, figsize: tuple[float, float] = [12, 6], limit_left: str = None, limit_right: str = None):
    # Prepare data for the first dataframe
    daily_incidents1 = dataframe1.groupby(dataframe1["time_call_received"].dt.date).size()

    # Prepare data for the second dataframe
    daily_incidents2 = dataframe2.groupby(dataframe2["time_call_received"].dt.date).size()

    # Plot data
    plt.figure(figsize=figsize)

    # Plot the first dataframe in blue
    plt.plot(daily_incidents1.index, daily_incidents1.values, color='blue', label='Before removing wrong timestamps')

    # Overlay the second dataframe in orange
    plt.plot(daily_incidents2.index, daily_incidents2.values, 'g--', label='After removing wrong timestamps')

    # Aesthetics
    if limit_left is not None:
        plt.xlim(left=pd.Timestamp(limit_left))
    if limit_right is not None:
        plt.xlim(right=pd.Timestamp(limit_right))

    plt.gca().xaxis.set_major_locator(matplotlib.dates.YearLocator())
    plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))

    plt.grid(True)

    # Set labels and title
    plt.title("Total Incidents Per Day Over Years")
    plt.xlabel("Year")
    plt.ylabel("Total Incidents")
    plt.legend()

    # Save/show plot
    plt.show()


def plot_time_difference_distribution(
    dataframe: pd.DataFrame,
    column_start: str,
    column_end: str,
    triage_impression: str = None,
    log_scale: bool = False
):
    """
    Plots the distribution of time differences in seconds between two datetime columns,
    excluding rows with None/NaT values, with an option to use a logarithmic scale for the y-axis.

    Parameters:
    - dataframe: pd.DataFrame containing the data.
    - column_start: The name of the start time column.
    - column_end: The name of the end time column.
    - log_scale: If True, use a logarithmic scale for the y-axis.
    """
    valid_rows = dataframe[column_start].notnull() & dataframe[column_end].notnull()
    if triage_impression is not None:
        valid_rows &= (dataframe["triage_impression_during_call"] != triage_impression)

    # calculate time difference in seconds only for rows without None/NaT values
    time_diffs = (dataframe.loc[valid_rows, column_end] - dataframe.loc[valid_rows, column_start]).dt.total_seconds()

    # plot the distribution of time differences
    matplotlib.pyplot.figure(figsize=(10, 6))
    matplotlib.pyplot.hist(time_diffs, bins=100, color="blue", edgecolor="black", alpha=0.7, log=log_scale)

    matplotlib.pyplot.title(f"Distribution of Time Differences ({column_start} to {column_end})")
    matplotlib.pyplot.xlabel("Time Difference in Seconds")
    matplotlib.pyplot.ylabel("Frequency" if not log_scale else "Log(Frequency)")
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.show()

    print(f"Mean time difference: {time_diffs.mean()} seconds")
    print(f"Median time difference: {time_diffs.median()} seconds")
    print(f"Standard deviation of time difference: {time_diffs.std()} seconds")
    print(f"Maximum time difference: {time_diffs.max()} seconds")
    print(f"Minimum time difference: {time_diffs.min()} seconds")
