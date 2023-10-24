import constants

import os
import pandas as pd
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


def latlon_to_utm(lat, lon, false_easting=0):
    wgs84 = pyproj.Proj(proj="latlong", datum="WGS84")
    utm33 = pyproj.Proj(proj="utm", zone=33, datum="WGS84")
    utm_x, utm_y = pyproj.transform(wgs84, utm33, lon, lat)
    utm_x += false_easting  # Add false easting if specified
    return utm_x, utm_y


def utm_to_latlon(utm_x, utm_y, false_easting=0):
    wgs84 = pyproj.Proj(proj="latlong", datum="WGS84")
    utm33 = pyproj.Proj(proj="utm", zone=33, datum="WGS84")
    utm_x -= false_easting  # Remove false easting if specified
    lon, lat = pyproj.transform(utm33, wgs84, utm_x, utm_y)
    return lat, lon
