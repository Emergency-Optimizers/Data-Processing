import constants

import os
import pyproj
import math
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


# Create a transformer to convert from geographic to UTM
transformer_geographic_to_utm = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:326{33}")
# Create a transformer to convert from UTM to geographic
transformer_utm_to_geographic = pyproj.Transformer.from_crs(f"EPSG:326{33}", "EPSG:4326")


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


def geographic_to_utm(longitude, latitude, zone=33, offset=0):
    """
    Convert geographic (longitude and latitude) coordinates to UTM coordinates.

    Parameters:
    - longitude (float): Longitude of the geographic coordinate.
    - latitude (float): Latitude of the geographic coordinate.
    - zone (int, optional): UTM zone number for the conversion. Default is 33, which covers parts of Norway.
    - offset (int, optional): False easting value. This shifts the UTM x-coordinate by the specified value.
    
    Returns:
    - tuple: UTM x-coordinate and UTM y-coordinate.
    """
    utm_easting, utm_northing = transformer_geographic_to_utm.transform(latitude, longitude)
    utm_easting += offset
    return utm_easting, utm_northing


def utm_to_geographic(easting, northing, zone=33, offset=0):
    """
    Convert UTM coordinates to geographic (longitude and latitude) coordinates.

    Parameters:
    - easting (float): UTM x-coordinate.
    - northing (float): UTM y-coordinate.
    - zone (int, optional): UTM zone number for the conversion. Default is 33, which covers parts of Norway.
    - offset (int, optional): False easting value. This subtracts the specified value from the UTM x-coordinate before conversion.
    
    Returns:
    - tuple: Longitude and latitude of the geographic coordinate.
    """
    easting -= offset
    longitude, latitude = transformer_utm_to_geographic.transform(easting, northing)
    return longitude, latitude


def utm_to_id(easting, northing):
    """Convert easting and northing to an ID based on the given formula."""
    ID = 2 * (10**13) + easting * (10**7) + northing
    return ID


def id_to_row_col(grid_id, cell_size=1000):
    """
    Convert grid ID to row and column values.
    
    Parameters:
    - grid_id: int
        The ID of the grid cell.
    - cell_size: int
        The height (and potentially width if the grid is square) of each cell in meters.
    
    Returns:
    - row: int
        Row index of the grid cell.
    - col: int
        Column index of the grid cell.
    """
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
    """
    Convert row and col values to grid ID.

    Parameters:
    - row: int
        Row index of the grid cell.
    - col: int
        Column index of the grid cell.
    - cell_size: int
        The height (and width, since cells are quadratic) of each cell in meters.

    Returns:
    - grid_id: int
        The ID of the grid cell.
    """
    
    # Convert row and col to northing and easting
    Y_c = row * cell_size
    X_c = col * cell_size

    # Compute grid ID using the formula
    grid_id = 2 * 10**13 + X_c * 10**7 + Y_c

    return grid_id


def id_to_easting_northing(grid_id):
    """
    Convert grid ID to easting and northing values.
    
    Parameters:
    - grid_id: int
        The ID of the grid cell.
    
    Returns:
    - easting: int
        Easting (X_c) value of the grid cell's southwestern corner.
    - northing: int
        Northing (Y_c) value of the grid cell's southwestern corner.
    """

    # Extracting Y_c (northing) from the ID
    northing = grid_id % (10**7)
    
    # Extracting X_c (easting) from the ID
    easting = (grid_id - 2 * 10**13 - northing) // 10**7

    return easting, northing


def snap_utm_to_ssb_grid(easting, northing, cell_size=1000):
    return (
        (math.floor((easting + 2000000) / cell_size) * cell_size - 2000000),
        (math.floor(northing / cell_size) * cell_size)
    )


def centroid_to_ssb_grid_points(easting, northing):
    x_c, y_c = snap_utm_to_ssb_grid(easting, northing)
    ssb_grid_points = [(x_c, y_c)]

    for x_offset, y_offset in [(1000, 0), (1000, 1000), (0, 1000)]:
        ssb_grid_points.append((x_c + x_offset, y_c + y_offset))

    ssb_grid_points.append((x_c, y_c))
    return ssb_grid_points


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
    south_west_easting, south_west_northing = snap_utm_to_ssb_grid(easting, northing, cell_size)
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


def get_oslo_akershus_grid() -> pd.DataFrame:
    df_oslo = pd.read_csv(os.path.join(constants.PROJECT_DIRECTORY_PATH, "data", "ssb_2019_oslo_epsg32633.csv"), encoding="utf-8")
    df_akershus = pd.read_csv(os.path.join(constants.PROJECT_DIRECTORY_PATH, "data", "ssb_2019_akershus_epsg32633.csv"), encoding="utf-8")
    df = pd.concat([df_oslo, df_akershus])

    return df


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
