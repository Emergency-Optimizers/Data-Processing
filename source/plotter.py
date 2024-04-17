import constants

import pandas as pd
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np


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
    log_scale: bool = False,
    cancelled: bool = False,
    percentage_threshold: float = None
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
        valid_rows &= (dataframe["triage_impression_during_call"] == triage_impression)
    
    if cancelled:
        valid_rows &= (dataframe["time_ambulance_dispatch_to_hospital"].isna())

    # calculate time difference in seconds only for rows without None/NaT values
    time_diffs = (dataframe.loc[valid_rows, column_end] - dataframe.loc[valid_rows, column_start]).dt.total_seconds() / 60

    # plot the distribution of time differences
    matplotlib.pyplot.figure(figsize=(12, 6))
    matplotlib.pyplot.hist(time_diffs, bins=300, color="blue", edgecolor="black", alpha=0.7, log=log_scale)

    matplotlib.pyplot.title(f"Distribution of Time Differences ({column_start} to {column_end})")
    matplotlib.pyplot.xlabel("Time Difference in Minutes")
    matplotlib.pyplot.ylabel("Frequency" if not log_scale else "Log(Frequency)")
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.show()

    print(f"Mean time difference: {time_diffs.mean()} minutes")
    print(f"Median time difference: {time_diffs.median()} minutes")
    print(f"Standard deviation of time difference: {time_diffs.std()} minutes")
    print(f"Maximum time difference: {time_diffs.max()} minutes")
    print(f"Minimum time difference: {time_diffs.min()} minutes")

    if percentage_threshold is not None:
        below_threshold_percentage = (time_diffs <= percentage_threshold).mean() * 100
        print(f"Percentage of time differences below {percentage_threshold} minutes: {below_threshold_percentage:.2f}%")


def boxplot_time_at_steps(
    dataframe: pd.DataFrame,
    triage_impressions: list = None,
    bounds: tuple[str, str] = None,
    group_width: float = 1.3,
    step_gap: float = 0.8,
    margin: float = 0.1
):
    title = "Time Taken At Each Step of the Incident by Triage Category"
    triage_types = ["A", "H", "V1"][::-1]
    triage_colors = {"A": "red", "H": "blue", "V1": "green"}

    if bounds is not None:
        start_bound, end_bound = pd.to_datetime(bounds[0]), pd.to_datetime(bounds[1])
        df_filtered = dataframe[(dataframe['time_call_received'] >= start_bound) &
                                (dataframe['time_call_received'] <= end_bound)].copy()
    else:
        df_filtered = dataframe.copy()

    steps = {
        "At Hospital": ("time_ambulance_arrived_at_hospital", "time_ambulance_available"),
        "Dispatching to Hospital": ("time_ambulance_dispatch_to_hospital", "time_ambulance_arrived_at_hospital"),
        "At Scene": ("time_ambulance_arrived_at_scene", "time_ambulance_dispatch_to_hospital", "time_ambulance_available"),
        "Dispatching to Scene": ("time_ambulance_dispatch_to_scene", "time_ambulance_arrived_at_scene"),
        "Resource to Start Task": ("time_resource_appointed", "time_ambulance_dispatch_to_scene"),
        "Appointing Resource": ("time_incident_created", "time_resource_appointed"),
        "Creating Incident": ("time_call_received", "time_incident_created"),
    }

    plt.figure(figsize=(12, 8))

    position = 0  # Initial position for the first group
    ytick_positions = []  # To store ytick positions for labeling
    handles = []  # To store handles for the legend
    displayed_triage_types = []

    for step, times in steps.items():
        data_for_step = []  # Collects all boxplots' data for current step
        positions = [position + i * ((group_width + margin) / len(triage_types)) for i in range(len(triage_types))]
        ytick_positions.append(sum(positions) / len(positions))  # Middle position for labels
        triage_types_used = []

        for i, triage in enumerate(triage_types):
            if triage_impressions is not None and triage not in triage_impressions:
                continue
            else:
                triage_types_used.append(triage)

            temp_df = df_filtered[df_filtered["triage_impression_during_call"] == triage].copy()
            if len(times) == 3:
                temp_df.loc[temp_df[times[1]].isna(), 'Duration'] = (temp_df[times[2]] - temp_df[times[0]]).dt.total_seconds() / 60
                temp_df.loc[~temp_df[times[1]].isna(), 'Duration'] = (temp_df[times[1]] - temp_df[times[0]]).dt.total_seconds() / 60
            else:
                temp_df.loc[:, 'Duration'] = (temp_df[times[1]] - temp_df[times[0]]).dt.total_seconds() / 60
            data_for_step.append(temp_df['Duration'].dropna())
            displayed_triage_types.append(triage)

        # Plotting each triage type for the current step
        for i, data in enumerate(data_for_step):
            box = plt.boxplot(
                data,
                positions=[positions[i]],
                widths=group_width / len(triage_types) - margin,
                patch_artist=True,
                vert=False,
                boxprops=dict(facecolor=triage_colors[triage_types_used[i]]),
                showfliers=False
            )
            if step == list(steps.keys())[0]:  # Only add to legend once per triage type displayed
                handles.append(box["boxes"][0])

        position += group_width + step_gap

    # Customizing the plot
    plt.title(title)
    plt.xlabel("Time in Minutes")
    plt.yticks(ytick_positions, [f"{step}" for step in steps])
    plt.legend(handles, list(dict.fromkeys(displayed_triage_types)), title="Triage", loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_percentage_below_threshold_per_hour(
    dataframe: pd.DataFrame,
    column_start: str,
    column_end: str,
    threshold: float,
    triage_impression: str = None,
    cancelled: bool = False
):
    """
    Plots the percentage of time differences below a specified threshold for each hour of the day,
    with options to filter rows based on triage impression and cancelled status.

    Parameters:
    - dataframe: pd.DataFrame containing the data.
    - column_start: The name of the start time column.
    - column_end: The name of the end time column.
    - threshold: The threshold (in minutes) to calculate the percentage for.
    - triage_impression: If not None, exclude rows with this value in the triage_impression_during_call column.
    - cancelled: If True, only consider rows where the time from ambulance dispatch to hospital is missing.
    """
    valid_rows = dataframe[column_start].notnull() & dataframe[column_end].notnull()
    if triage_impression is not None:
        valid_rows &= (dataframe["triage_impression_during_call"] == triage_impression)
    
    if cancelled:
        valid_rows &= (dataframe["time_ambulance_dispatch_to_hospital"].isna())

    dataframe = dataframe[valid_rows]
    time_diffs = (dataframe[column_end] - dataframe[column_start]).dt.total_seconds() / 60

    # Extract hour from the start time
    hours = dataframe[column_start].dt.hour

    percentages = []
    for hour in range(24):
        # Filter time differences by hour
        hourly_time_diffs = time_diffs[hours == hour]
        if len(hourly_time_diffs) > 0:
            # Calculate percentage below threshold for the hour
            percentage_below_threshold = (hourly_time_diffs <= threshold).mean() * 100
            percentages.append(percentage_below_threshold)
        else:
            percentages.append(0)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar(range(24), percentages, color="skyblue", edgecolor="black")
    plt.title(f"Percentage of Time Differences Below {threshold} Minutes for Each Hour of the Day")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Percentage Below Threshold")
    plt.xticks(range(24))
    plt.grid(axis='y')
    plt.show()
