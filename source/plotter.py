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
