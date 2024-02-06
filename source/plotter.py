import constants

import pandas as pd
import matplotlib.dates
import matplotlib.pyplot


def incidents_over_years(dataframe: pd.DataFrame, figsize: tuple[float, float] = [12, 6], limit_left: str = None, limit_right: str = None):
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


def plot_time_difference_distribution(dataframe: pd.DataFrame, column_start: str, column_end: str, log_scale: bool = False, IQR_multiplier: float = 1.5) -> tuple[float, float]:
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

    # calculate and print IQR-based cutoffs and statistics for additional insights
    Q1 = time_diffs.quantile(0.25)
    Q3 = time_diffs.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(0, Q1 - IQR_multiplier * IQR)
    upper_bound = Q3 + IQR_multiplier * IQR

    print(f"Mean time difference: {time_diffs.mean()} seconds")
    print(f"Median time difference: {time_diffs.median()} seconds")
    print(f"Standard deviation of time difference: {time_diffs.std()} seconds")
    print(f"Maximum time difference: {time_diffs.max()} seconds")
    print(f"Minimum time difference: {time_diffs.min()} seconds")
    print(f"Suggested upper bound for dropping rows: {upper_bound} seconds")
    print(f"Suggested lower bound for dropping rows: {lower_bound} seconds")

    return lower_bound, upper_bound
