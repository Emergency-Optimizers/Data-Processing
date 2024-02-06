import constants

import os
import math
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import utm
import math


def get_raw_incidents_path(dataset_id: str) -> str:
    return os.path.join(constants.PROJECT_DIRECTORY_PATH, "data", "raw", dataset_id, "incidents.csv")


def get_raw_depots_path(dataset_id: str) -> str:
    return os.path.join(constants.PROJECT_DIRECTORY_PATH, "data", "raw", dataset_id, "depots.csv")


def get_clean_incidents_path(dataset_id: str) -> str:
    return os.path.join(constants.PROJECT_DIRECTORY_PATH, "data", "clean", dataset_id, "incidents.csv")


def get_clean_depots_path(dataset_id: str) -> str:
    return os.path.join(constants.PROJECT_DIRECTORY_PATH, "data", "clean", dataset_id, "depots.csv")


def get_processed_incidents_path(dataset_id: str) -> str:
    return os.path.join(constants.PROJECT_DIRECTORY_PATH, "data", "processed", dataset_id, "incidents.csv")


def get_processed_depots_path(dataset_id: str) -> str:
    return os.path.join(constants.PROJECT_DIRECTORY_PATH, "data", "processed", dataset_id, "depots.csv")


def get_enhanced_incidents_path(dataset_id: str) -> str:
    return os.path.join(constants.PROJECT_DIRECTORY_PATH, "data", "enhanced", dataset_id, "incidents.csv")


def get_enhanced_depots_path(dataset_id: str) -> str:
    return os.path.join(constants.PROJECT_DIRECTORY_PATH, "data", "enhanced", dataset_id, "depots.csv")


def geographic_to_utm(lon, lat, zone=33):
    x, y, _, _, = utm.from_latlon(lat, lon, zone)
    return x, y


def utm_to_geographic(x, y, zone=33, northern=True):
    lat, lon = utm.to_latlon(x, y, zone, northern=northern, strict=False)
    return lon, lat


def utm_to_id(x, y, cell_size=1000, offset=2000000):
    x_corner = math.floor((x + offset) / cell_size) * cell_size - offset
    y_corner = (math.floor(y / cell_size) * cell_size)
    return 20000000000000 + (x_corner * 10000000) + y_corner


def id_to_utm(grid_id):
    x = math.floor(grid_id * (10**(-7))) - (2 * (10**6))
    y = grid_id - (math.floor(grid_id * (10**(-7))) * (10**7))
    return x, y


def id_to_row_col(grid_id, cell_size=1000):
    # extracting Y_c (northing) from the ID
    Y_c = grid_id % (10**7)
    # extracting X_c (easting) from the ID
    X_c = (grid_id - 2 * 10**13 - Y_c) // 10**7
    # convert Y_c (northing) to row index
    row = Y_c // cell_size
    # convert X_c (easting) to column index
    col = X_c // cell_size

    return row, col


def row_col_to_id(row, col, cell_size=1000):
    # convert row and col to northing and easting
    Y_c = row * cell_size
    X_c = col * cell_size
    # compute grid ID using the formula
    grid_id = 2 * 10**13 + X_c * 10**7 + Y_c

    return grid_id


def snap_utm_to_grid(x, y, cell_size=1000, offset=2000000):
    return (
        (math.floor((x + offset) / cell_size) * cell_size - offset),
        (math.floor(y / cell_size) * cell_size)
    )


def copy_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a deep copy of the given dataframe.

    Args:
    - df (pd.DataFrame): The original dataframe to be copied.

    Returns:
    - pd.DataFrame: A deep copy of the original dataframe.
    """
    return df.copy(deep=True)


def get_cell_corners(easting: int, northing: int, cell_size=1000) -> list[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    south_west_easting, south_west_northing = snap_utm_to_grid(easting, northing, cell_size)
    south_east_easting, south_east_northing = south_west_easting + cell_size, south_west_northing
    north_east_easting, north_east_northing = south_west_easting + cell_size, south_west_northing + cell_size
    north_west_easting, north_west_northing = south_west_easting, south_west_northing + cell_size

    corners = [
        utm_to_geographic(south_west_easting, south_west_northing),
        utm_to_geographic(south_east_easting, south_east_northing),
        utm_to_geographic(north_east_easting, north_east_northing),
        utm_to_geographic(north_west_easting, north_west_northing),
    ]
    return corners


def get_bounds(file_paths: list[str]) -> gpd.GeoDataFrame:
    gdfs = []

    for file_path in file_paths:
        gdfs.append(gpd.read_file(file_path))

    gdf_combined = pd.concat(gdfs, ignore_index=True)

    return gdf_combined


def plot_multiple_geojson_polygons_in_one_plot_corrected(file_paths, labels, colors):
    plt.figure(figsize=(15, 15))
    
    for idx, file_path in enumerate(file_paths):
        # Load the GeoJSON file into a GeoDataFrame
        gdf = gpd.read_file(file_path)
        
        # Plot the boundary of the GeoDataFrame
        gdf.boundary.plot(ax=plt.gca(), label=labels[idx], edgecolor=colors[idx])
        
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Polygons of Different Areas')
    plt.legend()
    plt.grid(True)
    plt.show()
