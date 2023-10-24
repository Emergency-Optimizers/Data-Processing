import constants

import os
import pyproj


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
    # Create a transformer to convert from geographic to UTM
    transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:326{zone}")
    utm_easting, utm_northing = transformer.transform(latitude, longitude)
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
    # Create a transformer to convert from UTM to geographic
    transformer = pyproj.Transformer.from_crs(f"EPSG:326{zone}", "EPSG:4326")
    longitude, latitude = transformer.transform(easting, northing)
    return longitude, latitude
