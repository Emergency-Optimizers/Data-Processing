import constants
import utils

import os
import pandas as pd
import numpy as np
import regex
from tqdm import tqdm
from scipy.stats import norm
import shapely.geometry
import math


class DataPreprocessor:
    """Abstract class for preprocessing datasets."""

    def __init__(self, dataset_id: str) -> None:
        """
        Initialize the data preprocessor.

        Args:
            dataset_id: Unique identifier for the dataset.
        """
        self.dataset_id = dataset_id
        # paths for raw data
        self._raw_incidents_data_path = utils.get_raw_incidents_path(self.dataset_id)
        self._raw_depots_data_path = utils.get_raw_depots_path(self.dataset_id)
        # paths for cleaned data
        self._clean_incidents_data_path = utils.get_clean_incidents_path(self.dataset_id)
        self._clean_depots_data_path = utils.get_clean_depots_path(self.dataset_id)
        # paths for processed data
        self._processed_incidents_data_path = utils.get_processed_incidents_path(self.dataset_id)
        self._processed_depots_data_path = utils.get_processed_depots_path(self.dataset_id)
        # paths for enhanced data
        self._enhanced_incidents_data_path = utils.get_enhanced_incidents_path(self.dataset_id)
        self._enhanced_depots_data_path = utils.get_enhanced_depots_path(self.dataset_id)

    def execute(self) -> None:
        """Run the preprocessor to clean and process the dataset."""
        if not os.path.exists(self._raw_incidents_data_path) or not os.path.exists(self._raw_depots_data_path):
            raise Exception("Missing the raw data files.")

        self._clean_data()
        self._process_data()
        self._enhance_data()

    def _clean_data(self) -> None:
        """Clean the raw data and cache it."""
        os.makedirs(os.path.dirname(self._clean_incidents_data_path), exist_ok=True)

        progress_bar = tqdm(desc="Cleaning dataset", total=2)

        if not os.path.exists(self._clean_incidents_data_path):
            self._clean_incidents()
        progress_bar.update(1)

        if not os.path.exists(self._clean_depots_data_path):
            self._clean_depots()
        progress_bar.update(1)

    def _clean_incidents(self) -> None:
        pass

    def _clean_depots(self) -> None:
        pass

    def _process_data(self) -> None:
        """Process the clean data and cache it."""
        os.makedirs(os.path.dirname(self._processed_incidents_data_path), exist_ok=True)

        progress_bar = tqdm(desc="Processing dataset", total=2)

        if not os.path.exists(self._processed_incidents_data_path):
            self._process_incidents()
        progress_bar.update(1)

        if not os.path.exists(self._processed_depots_data_path):
            self._process_depots()
        progress_bar.update(1)

    def _process_incidents(self) -> None:
        pass

    def _process_depots(self) -> None:
        pass

    def _enhance_data(self) -> None:
        """Enhance the processed data and cache it."""
        os.makedirs(os.path.dirname(self._enhanced_incidents_data_path), exist_ok=True)

        progress_bar = tqdm(desc="Enhancing dataset", total=2)

        if not os.path.exists(self._enhanced_incidents_data_path):
            self._enhance_incidents()
        progress_bar.update(1)

        if not os.path.exists(self._enhanced_depots_data_path):
            self._enhance_depots()
        progress_bar.update(1)

    def _enhance_incidents(self) -> None:
        pass

    def _enhance_depots(self) -> None:
        pass

    def save_dataframe(self, dataframe: pd.DataFrame, filepath: str):
        dataframe.to_csv(filepath, index=False)

    def load_clean_incidents_dataframe(self) -> pd.DataFrame:
        return pd.read_csv(self._clean_incidents_data_path, low_memory=False)

    def load_clean_depots_dataframe(self) -> pd.DataFrame:
        return pd.read_csv(self._clean_depots_data_path, low_memory=False)

    def load_processed_incidents_dataframe(self) -> pd.DataFrame:
        return pd.read_csv(self._processed_incidents_data_path, low_memory=False)

    def load_processed_depots_dataframe(self) -> pd.DataFrame:
        return pd.read_csv(self._processed_depots_data_path, low_memory=False)

    def load_enhanced_incidents_dataframe(self) -> pd.DataFrame:
        return pd.read_csv(self._enhanced_incidents_data_path, low_memory=False)

    def load_enhanced_depots_dataframe(self) -> pd.DataFrame:
        return pd.read_csv(self._enhanced_depots_data_path, low_memory=False)


class DataPreprocessorOUS_V2(DataPreprocessor):
    """Class for preprocessing the OUS dataset. Version 2.0"""

    def __init__(self) -> None:
        super().__init__(dataset_id="oslo")

    def _clean_incidents(self) -> None:
        self.fix_csv_errors()
        dataframe = pd.read_csv(self._clean_incidents_data_path, escapechar="\\", low_memory=False)
        dataframe = self.drop_unnecessary_raw_columns(dataframe)
        dataframe = self.fix_raw_types(dataframe)

        self.save_dataframe(dataframe, self._clean_incidents_data_path)

    def fix_csv_errors(self) -> None:
        """Fixes common errors in a CSV file and saves the corrected version."""
        with open(self._raw_incidents_data_path, "r", encoding="windows-1252") as source_file, \
             open(self._clean_incidents_data_path, "w", encoding="utf-8") as target_file:

            # fix empty header
            header = source_file.readline().replace('""', '"index"')
            target_file.write(header)

            # fix comma errors in the data lines
            for line in source_file:
                if regex.match(r".*\(.*,.*\).*", line):
                    line = regex.sub(r"\([^,()]+\K,", "\\,", line)

                target_file.write(line)

    def drop_unnecessary_raw_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        columns_to_drop = [
            "index",
            "utrykningstid",
            "responstid",
            "gml_id",
            "lokalId",
            "navnerom",
            "versjonId",
            "oppdateringsdato",
            "datauttaksdato",
            "opphav",
            "rsize",
            "col",
            "row",
            "statistikkÅr",
            "geometry"
        ]
        dataframe.drop(columns_to_drop, axis=1, inplace=True)

        return dataframe

    def fix_raw_types(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        headers_types = {
            "hastegrad": "object",
            "tidspunkt": "object",
            "tiltak_opprettet": "object",
            "ressurs_id": "object",
            "tiltak_type": "object",
            "varslet": "object",
            "rykker_ut": "object",
            "ank_hentested": "object",
            "avg_hentested": "object",
            "ank_levsted": "object",
            "ledig": "object",
            "ssbid1000M": "int64",
            "xcoor": "int64",
            "ycoor": "int64",
            "popTot": "int64",
            "popAve": "float64",
            "popFem": "int64",
            "popMal": "int64",
        }

        dataframe = dataframe.astype(headers_types)

        date_columns = [
            "tidspunkt",
            "tiltak_opprettet",
            "varslet",
            "rykker_ut",
            "ank_hentested",
            "avg_hentested",
            "ank_levsted",
            "ledig"
        ]

        # convert dates to correct pandas format
        for col in date_columns:
            dataframe[col] = pd.to_datetime(dataframe[col], format="%d.%m.%Y %H:%M:%S ", errors="coerce")

        return dataframe

    def _clean_depots(self) -> None:
        # nothing needed to be done
        dataframe = pd.read_csv(self._raw_depots_data_path)
        self.save_dataframe(dataframe, self._clean_depots_data_path)

    def _process_incidents(self) -> None:
        # defines dataframe according to template
        dataframe = self.initialize_processed_incidents_dataframe()
        dataframe = self.add_geo_data(dataframe)
        dataframe = self._count_resources_sent(dataframe)
        dataframe = self._count_total_per_hour_triage(dataframe)

        self.save_dataframe(dataframe, self._processed_incidents_data_path)

    def initialize_processed_incidents_dataframe(self) -> pd.DataFrame:
        dataframe_clean = self.load_clean_incidents_dataframe()

        dataframe = pd.DataFrame()
        dataframe["triage_impression_during_call"] = dataframe_clean["hastegrad"]
        dataframe["resource_id"] = dataframe_clean["ressurs_id"]
        dataframe["resource_type"] = dataframe_clean["tiltak_type"]
        dataframe["resources_sent"] = 0
        dataframe["time_call_received"] = dataframe_clean["tidspunkt"]
        dataframe["time_incident_created"] = dataframe_clean["tiltak_opprettet"]
        dataframe["time_resource_appointed"] = dataframe_clean["varslet"]
        dataframe["time_ambulance_dispatch_to_scene"] = dataframe_clean["rykker_ut"]
        dataframe["time_ambulance_arrived_at_scene"] = dataframe_clean["ank_hentested"]
        dataframe["time_ambulance_dispatch_to_hospital"] = dataframe_clean["avg_hentested"]
        dataframe["time_ambulance_arrived_at_hospital"] = dataframe_clean["ank_levsted"]
        dataframe["time_ambulance_available"] = dataframe_clean["ledig"]
        dataframe["grid_id"] = dataframe_clean["ssbid1000M"]
        dataframe["x"] = dataframe_clean["xcoor"]
        dataframe["y"] = dataframe_clean["ycoor"]
        dataframe["longitude"] = np.nan
        dataframe["latitude"] = np.nan
        dataframe["region"] = None
        dataframe["urban_settlement_ssb"] = False
        dataframe["urban_settlement_fhi"] = False
        dataframe["total_A_incidents_hour_0"] = 0
        dataframe["total_A_incidents_hour_1"] = 0
        dataframe["total_A_incidents_hour_2"] = 0
        dataframe["total_A_incidents_hour_3"] = 0
        dataframe["total_A_incidents_hour_4"] = 0
        dataframe["total_A_incidents_hour_5"] = 0
        dataframe["total_A_incidents_hour_6"] = 0
        dataframe["total_A_incidents_hour_7"] = 0
        dataframe["total_A_incidents_hour_8"] = 0
        dataframe["total_A_incidents_hour_9"] = 0
        dataframe["total_A_incidents_hour_10"] = 0
        dataframe["total_A_incidents_hour_11"] = 0
        dataframe["total_A_incidents_hour_12"] = 0
        dataframe["total_A_incidents_hour_13"] = 0
        dataframe["total_A_incidents_hour_14"] = 0
        dataframe["total_A_incidents_hour_15"] = 0
        dataframe["total_A_incidents_hour_16"] = 0
        dataframe["total_A_incidents_hour_17"] = 0
        dataframe["total_A_incidents_hour_18"] = 0
        dataframe["total_A_incidents_hour_19"] = 0
        dataframe["total_A_incidents_hour_20"] = 0
        dataframe["total_A_incidents_hour_21"] = 0
        dataframe["total_A_incidents_hour_22"] = 0
        dataframe["total_A_incidents_hour_23"] = 0
        dataframe["total_H_incidents_hour_0"] = 0
        dataframe["total_H_incidents_hour_1"] = 0
        dataframe["total_H_incidents_hour_2"] = 0
        dataframe["total_H_incidents_hour_3"] = 0
        dataframe["total_H_incidents_hour_4"] = 0
        dataframe["total_H_incidents_hour_5"] = 0
        dataframe["total_H_incidents_hour_6"] = 0
        dataframe["total_H_incidents_hour_7"] = 0
        dataframe["total_H_incidents_hour_8"] = 0
        dataframe["total_H_incidents_hour_9"] = 0
        dataframe["total_H_incidents_hour_10"] = 0
        dataframe["total_H_incidents_hour_11"] = 0
        dataframe["total_H_incidents_hour_12"] = 0
        dataframe["total_H_incidents_hour_13"] = 0
        dataframe["total_H_incidents_hour_14"] = 0
        dataframe["total_H_incidents_hour_15"] = 0
        dataframe["total_H_incidents_hour_16"] = 0
        dataframe["total_H_incidents_hour_17"] = 0
        dataframe["total_H_incidents_hour_18"] = 0
        dataframe["total_H_incidents_hour_19"] = 0
        dataframe["total_H_incidents_hour_20"] = 0
        dataframe["total_H_incidents_hour_21"] = 0
        dataframe["total_H_incidents_hour_22"] = 0
        dataframe["total_H_incidents_hour_23"] = 0
        dataframe["total_V1_incidents_hour_0"] = 0
        dataframe["total_V1_incidents_hour_1"] = 0
        dataframe["total_V1_incidents_hour_2"] = 0
        dataframe["total_V1_incidents_hour_3"] = 0
        dataframe["total_V1_incidents_hour_4"] = 0
        dataframe["total_V1_incidents_hour_5"] = 0
        dataframe["total_V1_incidents_hour_6"] = 0
        dataframe["total_V1_incidents_hour_7"] = 0
        dataframe["total_V1_incidents_hour_8"] = 0
        dataframe["total_V1_incidents_hour_9"] = 0
        dataframe["total_V1_incidents_hour_10"] = 0
        dataframe["total_V1_incidents_hour_11"] = 0
        dataframe["total_V1_incidents_hour_12"] = 0
        dataframe["total_V1_incidents_hour_13"] = 0
        dataframe["total_V1_incidents_hour_14"] = 0
        dataframe["total_V1_incidents_hour_15"] = 0
        dataframe["total_V1_incidents_hour_16"] = 0
        dataframe["total_V1_incidents_hour_17"] = 0
        dataframe["total_V1_incidents_hour_18"] = 0
        dataframe["total_V1_incidents_hour_19"] = 0
        dataframe["total_V1_incidents_hour_20"] = 0
        dataframe["total_V1_incidents_hour_21"] = 0
        dataframe["total_V1_incidents_hour_22"] = 0
        dataframe["total_V1_incidents_hour_23"] = 0

        dataframe = dataframe.sort_values(by="time_call_received")

        return dataframe

    def _process_depots(self) -> None:
        dataframe = self.initialize_processed_depots_dataframe()
        dataframe = self.convert_depot_types(dataframe)
        dataframe = self.add_grid_id(dataframe)
        dataframe = self.add_geo_data(dataframe)
        dataframe = self._add_population_data(dataframe, use_enhanced=False)

        self.save_dataframe(dataframe, self._processed_depots_data_path)

    def initialize_processed_depots_dataframe(self) -> pd.DataFrame:
        dataframe_clean = self.load_clean_depots_dataframe()

        dataframe = pd.DataFrame()
        dataframe["name"] = dataframe_clean["name"]
        dataframe["type"] = dataframe_clean["type"]
        dataframe["grid_id"] = 0
        dataframe["x"] = dataframe_clean["easting"]
        dataframe["y"] = dataframe_clean["northing"]
        dataframe["longitude"] = np.nan
        dataframe["latitude"] = np.nan
        dataframe["region"] = None
        dataframe["urban_settlement_ssb"] = False
        dataframe["urban_settlement_fhi"] = False
        dataframe["total_population_radius_2km"] = 0
        dataframe["total_population_radius_5km"] = 0
        dataframe["total_incidents_radius_2km"] = 0
        dataframe["total_incidents_radius_5km"] = 0
        dataframe["total_population_cluster"] = 0
        dataframe["total_incidents_cluster"] = 0

        return dataframe

    def convert_depot_types(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        for index, _ in dataframe.iterrows():
            depot_type = dataframe.at[index, "type"]

            new_depot_type = None
            if depot_type == "stasjon":
                new_depot_type = "Depot"
            elif depot_type == "beredskapspunkt":
                new_depot_type = "Beredskapspunkt"
            elif depot_type == "sykehus":
                new_depot_type = "Hospital"
            elif depot_type == "legevakt":
                new_depot_type = "Emergency Ward"

            dataframe.at[index, "type"] = new_depot_type

        return dataframe

    def add_grid_id(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        for index, _ in dataframe.iterrows():
            x_accurate = dataframe.at[index, "x"]
            y_accurate = dataframe.at[index, "y"]

            grid_id = utils.utm_to_id(x_accurate, y_accurate)
            x, y = utils.id_to_utm(grid_id)

            dataframe.at[index, "grid_id"] = grid_id
            dataframe.at[index, "x"] = x
            dataframe.at[index, "y"] = y

        return dataframe

    def add_geo_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        gdf_oslo_bounds = utils.get_bounds(file_paths=[os.path.join(constants.PROJECT_DIRECTORY_PATH, "data", "ssb_2019_oslo_polygon_epsg4326.geojson")])
        gdf_akershus_bounds = utils.get_bounds(file_paths=[os.path.join(constants.PROJECT_DIRECTORY_PATH, "data", "ssb_2019_akershus_polygon_epsg4326.geojson")])
        gdf_urban_settlement_bounds = utils.get_bounds(file_paths=[os.path.join(constants.PROJECT_DIRECTORY_PATH, "data", "ssb_2021_urban_settlements_polygon_epsg4326.geojson")])
        gdf_urban_settlement_fhi_bounds = utils.get_bounds(file_paths=[os.path.join(constants.PROJECT_DIRECTORY_PATH, "data", "nonurban_municipalities.geojson")])

        cached_geo_data_point = {}
        cached_geo_data_grid_id = {}

        # add spatial data to each row and cache results for rows with same grid id
        for index, _ in dataframe.iterrows():
            grid_id = dataframe.at[index, "grid_id"]

            if grid_id in cached_geo_data_grid_id:
                longitude, latitude, region, urban_settlement_ssb, urban_settlement_fhi = cached_geo_data_grid_id[grid_id]
            else:
                x = dataframe.at[index, "x"]
                y = dataframe.at[index, "y"]

                region_akershus_count = 0
                region_oslo_count = 0
                urban_settlement_count = 0
                urban_settlement_count_fhi = 0

                for latitude, longitude in utils.get_cell_corners(x, y):
                    if (longitude, latitude) in cached_geo_data_point:
                        region, urban_settlement_ssb, urban_settlement_fhi = cached_geo_data_point[(longitude, latitude)]
                    else:
                        point = shapely.geometry.Point(longitude, latitude)

                        region = None
                        if gdf_akershus_bounds.contains(point).any():
                            region = "Akershus"
                        elif gdf_oslo_bounds.contains(point).any():
                            region = "Oslo"

                        urban_settlement_ssb = gdf_urban_settlement_bounds.contains(point).any()
                        urban_settlement_fhi = not gdf_urban_settlement_fhi_bounds.contains(point).any()

                        cached_geo_data_point[(longitude, latitude)] = (region, urban_settlement_ssb, urban_settlement_fhi)

                    if region == "Akershus":
                        region_akershus_count += 1
                    elif region == "Oslo":
                        region_oslo_count += 1

                    urban_settlement_count += urban_settlement_ssb
                    urban_settlement_count_fhi += urban_settlement_fhi

                if (region_akershus_count + region_oslo_count) == 0:
                    region = None
                elif region_oslo_count >= region_akershus_count:
                    region = "Oslo"
                else:
                    region = "Akershus"

                urban_settlement_ssb = urban_settlement_count >= 1
                urban_settlement_fhi = urban_settlement_count_fhi >= 1

                cached_geo_data_grid_id[grid_id] = (longitude, latitude, region, urban_settlement_ssb, urban_settlement_fhi)

            dataframe.at[index, "longitude"], dataframe.at[index, "latitude"] = utils.utm_to_geographic(
                dataframe.at[index, "x"],
                dataframe.at[index, "y"]
            )

            dataframe.at[index, "region"] = region
            dataframe.at[index, "urban_settlement_ssb"] = urban_settlement_ssb
            dataframe.at[index, "urban_settlement_fhi"] = urban_settlement_fhi

        return dataframe

    def _add_population_data(self, dataframe: pd.DataFrame, use_enhanced: bool) -> pd.DataFrame:
        clean_incidents_df = self.load_clean_incidents_dataframe()
        clean_incidents_df = clean_incidents_df.drop_duplicates(subset="ssbid1000M")
        utm_to_population = {tuple(x + 500 for x in utils.id_to_utm(grid_id)): pop for grid_id, pop in zip(clean_incidents_df["ssbid1000M"], clean_incidents_df["popTot"])}

        if use_enhanced:
            incidents_df = self.load_enhanced_incidents_dataframe()
        else:
            incidents_df = self.load_processed_incidents_dataframe()

        grid_count = incidents_df["grid_id"].value_counts()
        utm_to_incidents = {tuple(x + 500 for x in utils.id_to_utm(grid_id)): count for grid_id, count in grid_count.items()}

        # set radius data
        for index, _ in dataframe.iterrows():
            target_utm = (dataframe.at[index, "x"], dataframe.at[index, "y"])
            dataframe.at[index, "total_population_radius_2km"] = utils.get_values_within_radius(utm_to_population, target_utm, distance_km=2.0)
            dataframe.at[index, "total_population_radius_5km"] = utils.get_values_within_radius(utm_to_population, target_utm, distance_km=5.0)
            dataframe.at[index, "total_incidents_radius_2km"] = utils.get_values_within_radius(utm_to_incidents, target_utm, distance_km=2.0)
            dataframe.at[index, "total_incidents_radius_5km"] = utils.get_values_within_radius(utm_to_incidents, target_utm, distance_km=5.0)

        # set cluster data
        dataframe["total_population_cluster"] = 0
        dataframe["total_incidents_cluster"] = 0

        for incident_utm, population in utm_to_population.items():
            min_distance = float('inf')
            closest_depot_index = -1

            for depot_index, depot in dataframe.iterrows():
                if (depot["type"] != "Depot" and depot["type"] != "Beredskapspunkt"):
                    continue

                depot_utm = (depot["x"] + 500, depot["y"] + 500)

                distance = math.dist(incident_utm, depot_utm)

                if (closest_depot_index == -1 or distance < min_distance):
                    closest_depot_index = depot_index
                    min_distance = distance

            dataframe.at[closest_depot_index, "total_population_cluster"] += population

        for incident_utm, incidents in utm_to_incidents.items():
            min_distance = float('inf')
            closest_depot_index = -1

            for depot_index, depot in dataframe.iterrows():
                if (depot["type"] != "Depot" and depot["type"] != "Beredskapspunkt"):
                    continue

                depot_utm = (depot["x"] + 500, depot["y"] + 500)

                distance = math.dist(incident_utm, depot_utm)

                if (closest_depot_index == -1 or distance < min_distance):
                    closest_depot_index = depot_index
                    min_distance = distance

            dataframe.at[closest_depot_index, "total_incidents_cluster"] += incidents

        return dataframe

    def _enhance_incidents(self) -> None:
        dataframe = self.load_processed_incidents_dataframe()

        dataframe = self._remove_duplicates(dataframe)
        dataframe = self._remove_incomplete_years(dataframe)
        dataframe = self._remove_outside_region(dataframe)
        dataframe = self._remove_other_resource_types(dataframe)
        dataframe = self._count_resources_sent(dataframe)
        dataframe = self._remove_extra_resources(dataframe)
        dataframe = self._count_total_per_hour_triage(dataframe)
        dataframe = self._remove_other_triage_impressions(dataframe)
        dataframe = self._remove_wrong_timestamps(dataframe)
        dataframe = self._fix_timestamps(dataframe)
        dataframe = self._remove_na(dataframe)
        dataframe = self._remove_outliers(dataframe)

        dataframe = dataframe.sort_values(by="time_call_received")

        # converts pandas date time to string format
        dataframe["time_call_received"] = dataframe["time_call_received"].dt.strftime("%Y-%m-%d %H:%M:%S")

        self.save_dataframe(dataframe, self._enhanced_incidents_data_path)

    def _remove_other_resource_types(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe[dataframe["resource_type"] == "Ambulanse"]

        return dataframe

    def _count_resources_sent(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        unique_counts = dataframe.groupby(["time_call_received", "grid_id"])["resource_id"].nunique()

        dataframe["resources_sent"] = dataframe.set_index(["time_call_received", "grid_id"]).index.map(unique_counts)

        return dataframe

    def _remove_extra_resources(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        # columns to consider for counting NaNs
        time_columns = [
            "time_incident_created",
            "time_resource_appointed",
            "time_ambulance_dispatch_to_scene",
            "time_ambulance_arrived_at_scene",
            "time_ambulance_dispatch_to_hospital",
            "time_ambulance_arrived_at_hospital",
            "time_ambulance_available"
        ]

        # sort by the number of NaNs in time columns (ascending) and then by 'time_call_received' and 'grid_id'
        dataframe = dataframe.sort_values(
            by=time_columns + ["time_call_received", "grid_id"],
            ascending=[True] * len(time_columns) + [True, True],
            na_position="last"
        )

        # drop duplicates, keeping the first occurrence (which has fewer NaNs)
        dataframe = dataframe.drop_duplicates(subset=["time_call_received", "grid_id"], keep="first")

        return dataframe

    def _remove_incomplete_years(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe["year"] = dataframe["time_call_received"].dt.year
        dataframe = dataframe[(dataframe['year'] >= 2016) & (dataframe['year'] <= 2018)]
        dataframe = dataframe.drop(columns=["year"])

        return dataframe

    def _remove_duplicates(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe.drop_duplicates(inplace=True)

        return dataframe

    def _remove_outside_region(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.dropna(subset=["region"])

        return dataframe

    def _remove_other_triage_impressions(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe[(dataframe["triage_impression_during_call"] != "V2") & (dataframe["triage_impression_during_call"] != "V")]

        return dataframe

    def _remove_wrong_timestamps(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        datetime_columns = [
            "time_call_received", "time_incident_created", "time_resource_appointed",
            "time_ambulance_dispatch_to_scene", "time_ambulance_arrived_at_scene", "time_ambulance_dispatch_to_hospital",
            "time_ambulance_arrived_at_hospital", "time_ambulance_available"
        ]

        # start with all rows marked as valid
        valid_rows = pd.Series([True] * len(dataframe), index=dataframe.index)

        # iterate over the datetime column pairs, except the first pair
        for i in range(1, len(datetime_columns) - 1):
            first_col = datetime_columns[i]
            second_col = datetime_columns[i + 1]

            # mark rows as invalid where the first date is after the second date
            valid_rows &= ~(dataframe[first_col] > dataframe[second_col])

        # return a new dataframe excluding the rows with incorrect timestamps
        return dataframe[valid_rows]

    def _fix_timestamps(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        # add a 'date' column for grouping
        dataframe["date"] = dataframe["time_call_received"].dt.floor("D")

        # identify rows where 'time_call_received' is after 'time_incident_created'
        invalid_rows_mask = dataframe["time_call_received"] > dataframe["time_incident_created"]

        # calculate the mean difference for each day for valid rows
        valid_diffs = dataframe.loc[~invalid_rows_mask, ["date", "time_call_received", "time_incident_created"]]
        valid_diffs["time_diff"] = (valid_diffs["time_incident_created"] - valid_diffs["time_call_received"]).dt.total_seconds()
        daily_mean_diffs = valid_diffs.groupby("date")["time_diff"].mean()

        # map daily mean differences back to the original dataframe for invalid rows
        dataframe.loc[invalid_rows_mask, "mean_diff"] = dataframe.loc[invalid_rows_mask, "date"].map(daily_mean_diffs)

        # adjust 'time_call_received' for invalid rows
        adjust_seconds = pd.to_timedelta(dataframe.loc[invalid_rows_mask, "mean_diff"], unit="s")
        dataframe.loc[invalid_rows_mask, "time_call_received"] = dataframe.loc[invalid_rows_mask, "time_incident_created"] - adjust_seconds

        # clean up temporary columns
        dataframe = dataframe.drop(columns=["date", "mean_diff"])

        return dataframe

    def _remove_na(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.dropna(
            subset=["triage_impression_during_call", "time_resource_appointed", "time_ambulance_dispatch_to_scene", "time_ambulance_arrived_at_scene", "time_ambulance_available"]
        )

        mask1 = dataframe["time_ambulance_arrived_at_scene"].isna() & dataframe["time_ambulance_arrived_at_hospital"].notna()
        mask2 = dataframe["time_ambulance_dispatch_to_hospital"].isna() & dataframe["time_ambulance_arrived_at_hospital"].notna()
        mask3 = dataframe["time_ambulance_dispatch_to_hospital"].notna() & dataframe["time_ambulance_arrived_at_hospital"].isna()
        dataframe = dataframe[~(mask1 | mask2 | mask3)]

        return dataframe

    def _remove_outliers(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = self._drop_outside_bounds(
            dataframe,
            "time_call_received",
            "time_incident_created",
            triage_impression="A",
            z_score_threshold=3,
            bounds_to_use="z"
        )

        dataframe = self._drop_outside_bounds(
            dataframe,
            "time_incident_created",
            "time_resource_appointed",
            triage_impression="A",
            z_score_threshold=3,
            bounds_to_use="z"
        )

        dataframe = self._drop_outside_bounds(
            dataframe,
            "time_resource_appointed",
            "time_ambulance_dispatch_to_scene",
            triage_impression="A",
            z_score_threshold=3,
            bounds_to_use="z"
        )

        dataframe = self._drop_outside_bounds(
            dataframe,
            "time_ambulance_dispatch_to_scene",
            "time_ambulance_arrived_at_scene",
            triage_impression="A",
            z_score_threshold=3,
            bounds_to_use="z"
        )

        dataframe = self._drop_outside_bounds(
            dataframe,
            "time_ambulance_arrived_at_scene",
            "time_ambulance_dispatch_to_hospital",
            triage_impression="A",
            z_score_threshold=3,
            bounds_to_use="z"
        )

        dataframe = self._drop_outside_bounds(
            dataframe,
            "time_ambulance_dispatch_to_hospital",
            "time_ambulance_arrived_at_hospital",
            triage_impression="A",
            z_score_threshold=3,
            bounds_to_use="z"
        )

        dataframe = self._drop_outside_bounds(
            dataframe,
            "time_ambulance_arrived_at_hospital",
            "time_ambulance_available",
            triage_impression="A",
            z_score_threshold=3,
            bounds_to_use="z"
        )

        dataframe = self._drop_outside_bounds(
            dataframe,
            "time_ambulance_arrived_at_scene",
            "time_ambulance_available",
            cancelled=True,
            triage_impression="A",
            z_score_threshold=3,
            bounds_to_use="z"
        )

        # H incidents
        dataframe = self._drop_outside_bounds(
            dataframe,
            "time_call_received",
            "time_incident_created",
            triage_impression="H",
            z_score_threshold=3,
            bounds_to_use="z"
        )

        dataframe = self._drop_outside_bounds(
            dataframe,
            "time_incident_created",
            "time_resource_appointed",
            triage_impression="H",
            z_score_threshold=3,
            bounds_to_use="z"
        )

        dataframe = self._drop_outside_bounds(
            dataframe,
            "time_resource_appointed",
            "time_ambulance_dispatch_to_scene",
            triage_impression="H",
            z_score_threshold=3,
            bounds_to_use="z"
        )

        dataframe = self._drop_outside_bounds(
            dataframe,
            "time_ambulance_dispatch_to_scene",
            "time_ambulance_arrived_at_scene",
            triage_impression="H",
            z_score_threshold=3,
            bounds_to_use="z"
        )

        dataframe = self._drop_outside_bounds(
            dataframe,
            "time_ambulance_arrived_at_scene",
            "time_ambulance_dispatch_to_hospital",
            triage_impression="H",
            z_score_threshold=3,
            bounds_to_use="z"
        )

        dataframe = self._drop_outside_bounds(
            dataframe,
            "time_ambulance_dispatch_to_hospital",
            "time_ambulance_arrived_at_hospital",
            triage_impression="H",
            z_score_threshold=3,
            bounds_to_use="z"
        )

        dataframe = self._drop_outside_bounds(
            dataframe,
            "time_ambulance_arrived_at_hospital",
            "time_ambulance_available",
            triage_impression="H",
            z_score_threshold=3,
            bounds_to_use="z"
        )

        dataframe = self._drop_outside_bounds(
            dataframe,
            "time_ambulance_arrived_at_scene",
            "time_ambulance_available",
            cancelled=True,
            triage_impression="H",
            z_score_threshold=3,
            bounds_to_use="z"
        )

        # V1 incidents (includes a special case)
        dataframe = self._drop_outside_bounds(
            dataframe,
            "time_call_received",
            "time_incident_created",
            triage_impression="V1",
            z_score_threshold=3,
            bounds_to_use="z"
        )

        dataframe = self._drop_outside_bounds(
            dataframe,
            "time_incident_created",
            "time_resource_appointed",
            triage_impression="V1",
            z_score_threshold=2,  # special case, would otherwise include very large outliers
            bounds_to_use="z"
        )

        dataframe = self._drop_outside_bounds(
            dataframe,
            "time_resource_appointed",
            "time_ambulance_dispatch_to_scene",
            triage_impression="V1",
            z_score_threshold=3,
            bounds_to_use="z"
        )

        dataframe = self._drop_outside_bounds(
            dataframe,
            "time_ambulance_dispatch_to_scene",
            "time_ambulance_arrived_at_scene",
            triage_impression="V1",
            z_score_threshold=3,
            bounds_to_use="z"
        )

        dataframe = self._drop_outside_bounds(
            dataframe,
            "time_ambulance_arrived_at_scene",
            "time_ambulance_dispatch_to_hospital",
            triage_impression="V1",
            z_score_threshold=3,
            bounds_to_use="z"
        )

        dataframe = self._drop_outside_bounds(
            dataframe,
            "time_ambulance_dispatch_to_hospital",
            "time_ambulance_arrived_at_hospital",
            triage_impression="V1",
            z_score_threshold=3,
            bounds_to_use="z"
        )

        dataframe = self._drop_outside_bounds(
            dataframe,
            "time_ambulance_arrived_at_hospital",
            "time_ambulance_available",
            triage_impression="V1",
            z_score_threshold=3,
            bounds_to_use="z"
        )

        dataframe = self._drop_outside_bounds(
            dataframe,
            "time_ambulance_arrived_at_scene",
            "time_ambulance_available",
            cancelled=True,
            triage_impression="V1",
            z_score_threshold=3,
            bounds_to_use="z"
        )

        return dataframe

    def _drop_outside_bounds(
        self,
        dataframe: pd.DataFrame,
        column_start: str,
        column_end: str,
        triage_impression: str = None,
        z_score_threshold: float = 3,
        IQR_multiplier: float = 1.5,
        bounds_to_use: str = "z",
        cancelled: bool = False,
        verbose: bool = False
    ) -> pd.DataFrame:
        keep_mask = pd.Series(True, index=dataframe.index)

        valid_rows = dataframe[column_start].notnull() & dataframe[column_end].notnull()
        if triage_impression is not None:
            valid_rows &= (dataframe["triage_impression_during_call"] == triage_impression)

        if cancelled:
            valid_rows &= (dataframe["time_ambulance_dispatch_to_hospital"].isna())

        time_diffs = (dataframe.loc[valid_rows, column_end] - dataframe.loc[valid_rows, column_start]).dt.total_seconds()

        z_bounds, iqr_bounds = utils.time_difference_lower_upper_bounds(
            time_diffs,
            z_score_threshold,
            IQR_multiplier,
            verbose
        )

        match bounds_to_use:
            case "z":
                bounds = z_bounds
            case "iqr":
                bounds = iqr_bounds
            case _:
                print("ERROR: unknown bounds to use")
                return

        keep_mask = pd.Series(True, index=dataframe.index)

        valid_rows = dataframe[column_start].notnull() & dataframe[column_end].notnull()
        if triage_impression is not None:
            valid_rows &= (dataframe["triage_impression_during_call"] == triage_impression)

        if cancelled:
            valid_rows &= (dataframe["time_ambulance_dispatch_to_hospital"].isna())

        time_diffs = (dataframe.loc[valid_rows, column_end] - dataframe.loc[valid_rows, column_start]).dt.total_seconds()

        keep_mask.loc[valid_rows] = keep_mask.loc[valid_rows] & (time_diffs >= bounds[0]).values
        keep_mask.loc[valid_rows] = keep_mask.loc[valid_rows] & (time_diffs <= bounds[1]).values

        keep_mask = keep_mask.astype(bool)

        return dataframe[keep_mask]

    def _count_total_per_hour_triage(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        # extract hour and date from 'time_call_received'
        # count V2 as V1
        dataframe['triage'] = dataframe['triage_impression_during_call'].replace(['V2', 'V'], 'V1')
        dataframe['hour'] = dataframe['time_call_received'].dt.hour
        dataframe['date'] = dataframe['time_call_received'].dt.date

        # group by date, hour, and triage type, then count incidents
        counts = dataframe.groupby(['date', 'hour', 'triage']).size().reset_index(name='count')

        pivot_table = counts.pivot_table(index='date', columns=['triage', 'hour'], values='count').fillna(0)

        # create new column names based on triage type and hour
        pivot_table.columns = [f"total_{triage}_incidents_hour_{hour}" for triage, hour in pivot_table.columns]

        pivot_table.reset_index(inplace=True)

        # update the original dataframe
        for column in pivot_table.columns[1:]:
            # prepare a dictionary to map 'date' to the counts for this triage type and hour
            update_dict = pivot_table[['date', column]].set_index('date')[column].to_dict()

            dataframe[column] = dataframe.apply(lambda row: update_dict.get(row['date'], 0), axis=1)

        # Convert total incidents columns to nullable integer type
        incident_columns = [col for col in dataframe.columns if col.startswith('total_')]
        for col in incident_columns:
            dataframe[col] = dataframe[col].astype('Int64')

        # drop temporary columns
        dataframe.drop(columns=['triage', 'hour', 'date'], inplace=True)

        return dataframe

    def _enhance_depots(self) -> None:
        dataframe = self.load_processed_depots_dataframe()
        dataframe = self._add_population_data(dataframe, use_enhanced=True)

        self.save_dataframe(dataframe, self._enhanced_depots_data_path)

    def load_clean_incidents_dataframe(self) -> pd.DataFrame:
        column_types = {
            "hastegrad": "object",
            "ressurs_id": "object",
            "tiltak_type": "object",
            "ssbid1000M": "int64",
            "xcoor": "int64",
            "ycoor": "int64",
            "popTot": "int64",
            "popAve": "float64",
            "popFem": "int64",
            "popMal": "int64",
            "geometry_x": "int64",
            "geometry_y": "int64"
        }
        column_index_dates = [1, 2, 5, 6, 7, 8, 9, 10]

        dataframe = pd.read_csv(
            self._clean_incidents_data_path,
            dtype=column_types,
            na_values=[""],
            parse_dates=column_index_dates
        )

        return dataframe

    def load_clean_depots_dataframe(self) -> pd.DataFrame:
        column_types = {
            "name": "object",
            "type": "object",
            "easting": "int64",
            "northing": "int64"
        }

        dataframe = pd.read_csv(
            self._clean_depots_data_path,
            dtype=column_types,
            na_values=[""]
        )

        return dataframe

    def load_processed_incidents_dataframe(self) -> pd.DataFrame:
        column_types = {
            "triage_impression_during_call": "object",
            "resource_id": "object",
            "resource_type": "object",
            "resources_sent": "int64",
            "grid_id": "int64",
            "x": "int64",
            "y": "int64",
            "longitude": "float64",
            "latitude": "float64",
            "region": "object",
            "urban_settlement_ssb": "bool",
            "urban_settlement_fhi": "bool",
            "total_A_incidents_hour_0": "int64",
            "total_A_incidents_hour_1": "int64",
            "total_A_incidents_hour_2": "int64",
            "total_A_incidents_hour_3": "int64",
            "total_A_incidents_hour_4": "int64",
            "total_A_incidents_hour_5": "int64",
            "total_A_incidents_hour_6": "int64",
            "total_A_incidents_hour_7": "int64",
            "total_A_incidents_hour_8": "int64",
            "total_A_incidents_hour_9": "int64",
            "total_A_incidents_hour_10": "int64",
            "total_A_incidents_hour_11": "int64",
            "total_A_incidents_hour_12": "int64",
            "total_A_incidents_hour_13": "int64",
            "total_A_incidents_hour_14": "int64",
            "total_A_incidents_hour_15": "int64",
            "total_A_incidents_hour_16": "int64",
            "total_A_incidents_hour_17": "int64",
            "total_A_incidents_hour_18": "int64",
            "total_A_incidents_hour_19": "int64",
            "total_A_incidents_hour_20": "int64",
            "total_A_incidents_hour_21": "int64",
            "total_A_incidents_hour_22": "int64",
            "total_A_incidents_hour_23": "int64",
            "total_H_incidents_hour_0": "int64",
            "total_H_incidents_hour_1": "int64",
            "total_H_incidents_hour_2": "int64",
            "total_H_incidents_hour_3": "int64",
            "total_H_incidents_hour_4": "int64",
            "total_H_incidents_hour_5": "int64",
            "total_H_incidents_hour_6": "int64",
            "total_H_incidents_hour_7": "int64",
            "total_H_incidents_hour_8": "int64",
            "total_H_incidents_hour_9": "int64",
            "total_H_incidents_hour_10": "int64",
            "total_H_incidents_hour_11": "int64",
            "total_H_incidents_hour_12": "int64",
            "total_H_incidents_hour_13": "int64",
            "total_H_incidents_hour_14": "int64",
            "total_H_incidents_hour_15": "int64",
            "total_H_incidents_hour_16": "int64",
            "total_H_incidents_hour_17": "int64",
            "total_H_incidents_hour_18": "int64",
            "total_H_incidents_hour_19": "int64",
            "total_H_incidents_hour_20": "int64",
            "total_H_incidents_hour_21": "int64",
            "total_H_incidents_hour_22": "int64",
            "total_H_incidents_hour_23": "int64",
            "total_V1_incidents_hour_0": "int64",
            "total_V1_incidents_hour_1": "int64",
            "total_V1_incidents_hour_2": "int64",
            "total_V1_incidents_hour_3": "int64",
            "total_V1_incidents_hour_4": "int64",
            "total_V1_incidents_hour_5": "int64",
            "total_V1_incidents_hour_6": "int64",
            "total_V1_incidents_hour_7": "int64",
            "total_V1_incidents_hour_8": "int64",
            "total_V1_incidents_hour_9": "int64",
            "total_V1_incidents_hour_10": "int64",
            "total_V1_incidents_hour_11": "int64",
            "total_V1_incidents_hour_12": "int64",
            "total_V1_incidents_hour_13": "int64",
            "total_V1_incidents_hour_14": "int64",
            "total_V1_incidents_hour_15": "int64",
            "total_V1_incidents_hour_16": "int64",
            "total_V1_incidents_hour_17": "int64",
            "total_V1_incidents_hour_18": "int64",
            "total_V1_incidents_hour_19": "int64",
            "total_V1_incidents_hour_20": "int64",
            "total_V1_incidents_hour_21": "int64",
            "total_V1_incidents_hour_22": "int64",
            "total_V1_incidents_hour_23": "int64",
        }
        column_index_dates = [4, 5, 6, 7, 8, 9, 10, 11]

        dataframe = pd.read_csv(
            self._processed_incidents_data_path,
            dtype=column_types,
            na_values=[""],
            parse_dates=column_index_dates
        )

        return dataframe

    def load_processed_depots_dataframe(self) -> pd.DataFrame:
        column_types = {
            "name": "object",
            "type": "object",
            "grid_id": "int64",
            "x": "int64",
            "y": "int64",
            "longitude": "float64",
            "latitude": "float64",
            "region": "object",
            "urban_settlement_ssb": "bool",
            "urban_settlement_fhi": "bool",
            "total_population_radius_2km": "int64",
            "total_population_radius_5km": "int64",
            "total_incidents_radius_2km": "int64",
            "total_incidents_radius_5km": "int64",
            "total_population_cluster": "int64",
            "total_incidents_cluster": "int64"
        }

        dataframe = pd.read_csv(
            self._processed_depots_data_path,
            dtype=column_types,
            na_values=[""]
        )

        return dataframe

    def load_enhanced_incidents_dataframe(self) -> pd.DataFrame:
        column_types = {
            "triage_impression_during_call": "object",
            "resource_id": "object",
            "resource_type": "object",
            "resources_sent": "int64",
            "grid_id": "int64",
            "x": "int64",
            "y": "int64",
            "longitude": "float64",
            "latitude": "float64",
            "region": "object",
            "urban_settlement_ssb": "bool",
            "urban_settlement_fhi": "bool",
            "total_A_incidents_hour_0": "int64",
            "total_A_incidents_hour_1": "int64",
            "total_A_incidents_hour_2": "int64",
            "total_A_incidents_hour_3": "int64",
            "total_A_incidents_hour_4": "int64",
            "total_A_incidents_hour_5": "int64",
            "total_A_incidents_hour_6": "int64",
            "total_A_incidents_hour_7": "int64",
            "total_A_incidents_hour_8": "int64",
            "total_A_incidents_hour_9": "int64",
            "total_A_incidents_hour_10": "int64",
            "total_A_incidents_hour_11": "int64",
            "total_A_incidents_hour_12": "int64",
            "total_A_incidents_hour_13": "int64",
            "total_A_incidents_hour_14": "int64",
            "total_A_incidents_hour_15": "int64",
            "total_A_incidents_hour_16": "int64",
            "total_A_incidents_hour_17": "int64",
            "total_A_incidents_hour_18": "int64",
            "total_A_incidents_hour_19": "int64",
            "total_A_incidents_hour_20": "int64",
            "total_A_incidents_hour_21": "int64",
            "total_A_incidents_hour_22": "int64",
            "total_A_incidents_hour_23": "int64",
            "total_H_incidents_hour_0": "int64",
            "total_H_incidents_hour_1": "int64",
            "total_H_incidents_hour_2": "int64",
            "total_H_incidents_hour_3": "int64",
            "total_H_incidents_hour_4": "int64",
            "total_H_incidents_hour_5": "int64",
            "total_H_incidents_hour_6": "int64",
            "total_H_incidents_hour_7": "int64",
            "total_H_incidents_hour_8": "int64",
            "total_H_incidents_hour_9": "int64",
            "total_H_incidents_hour_10": "int64",
            "total_H_incidents_hour_11": "int64",
            "total_H_incidents_hour_12": "int64",
            "total_H_incidents_hour_13": "int64",
            "total_H_incidents_hour_14": "int64",
            "total_H_incidents_hour_15": "int64",
            "total_H_incidents_hour_16": "int64",
            "total_H_incidents_hour_17": "int64",
            "total_H_incidents_hour_18": "int64",
            "total_H_incidents_hour_19": "int64",
            "total_H_incidents_hour_20": "int64",
            "total_H_incidents_hour_21": "int64",
            "total_H_incidents_hour_22": "int64",
            "total_H_incidents_hour_23": "int64",
            "total_V1_incidents_hour_0": "int64",
            "total_V1_incidents_hour_1": "int64",
            "total_V1_incidents_hour_2": "int64",
            "total_V1_incidents_hour_3": "int64",
            "total_V1_incidents_hour_4": "int64",
            "total_V1_incidents_hour_5": "int64",
            "total_V1_incidents_hour_6": "int64",
            "total_V1_incidents_hour_7": "int64",
            "total_V1_incidents_hour_8": "int64",
            "total_V1_incidents_hour_9": "int64",
            "total_V1_incidents_hour_10": "int64",
            "total_V1_incidents_hour_11": "int64",
            "total_V1_incidents_hour_12": "int64",
            "total_V1_incidents_hour_13": "int64",
            "total_V1_incidents_hour_14": "int64",
            "total_V1_incidents_hour_15": "int64",
            "total_V1_incidents_hour_16": "int64",
            "total_V1_incidents_hour_17": "int64",
            "total_V1_incidents_hour_18": "int64",
            "total_V1_incidents_hour_19": "int64",
            "total_V1_incidents_hour_20": "int64",
            "total_V1_incidents_hour_21": "int64",
            "total_V1_incidents_hour_22": "int64",
            "total_V1_incidents_hour_23": "int64",
        }
        column_index_dates = [4, 5, 6, 7, 8, 9, 10, 11]

        dataframe = pd.read_csv(
            self._enhanced_incidents_data_path,
            dtype=column_types,
            na_values=[""],
            parse_dates=column_index_dates
        )

        return dataframe

    def load_enhanced_depots_dataframe(self) -> pd.DataFrame:
        column_types = {
            "name": "object",
            "type": "object",
            "grid_id": "int64",
            "x": "int64",
            "y": "int64",
            "longitude": "float64",
            "latitude": "float64",
            "region": "object",
            "urban_settlement_ssb": "bool",
            "urban_settlement_fhi": "bool",
            "total_population_radius_2km": "int64",
            "total_population_radius_5km": "int64",
            "total_incidents_radius_2km": "int64",
            "total_incidents_radius_5km": "int64",
            "total_population_cluster": "int64",
            "total_incidents_cluster": "int64"
        }

        dataframe = pd.read_csv(
            self._enhanced_depots_data_path,
            dtype=column_types,
            na_values=[""]
        )

        return dataframe


class DataPreprocessorOUS(DataPreprocessor):
    """Class for preprocessing the OUS dataset. This is an older version, no longer used."""

    def __init__(self) -> None:
        super().__init__(dataset_id="oslo")

    def _clean_incidents(self) -> None:
        self._fix_csv()
        self._clean_and_save_incidents()

    def _clean_depots(self) -> None:
        df_depots = pd.read_csv(self._raw_depots_data_path)
        df_depots.to_csv(self._clean_depots_data_path, index=False)

    def _fix_csv(self) -> None:
        with open(self._raw_incidents_data_path, "r", encoding="windows-1252") as input_file, \
             open(self._clean_incidents_data_path, "w", encoding="utf-8") as output_file:

            header = input_file.readline().replace('""', '"id"')
            output_file.write(header)

            for line in input_file:
                if regex.match(r".*\(.*,.*\).*", line):
                    line = regex.sub(r"\([^,()]+\K,", "\\,", line)
                output_file.write(line)

    def _clean_and_save_incidents(self) -> None:
        df_incidents = pd.read_csv(self._clean_incidents_data_path, escapechar="\\", low_memory=False)
        # split geometry
        df_incidents[['real_x', 'real_y']] = df_incidents['geometry'].str.replace('c\(|\)', '', regex=True).str.split(', ', expand=True)
        # drop unnecessary columns
        columns_to_drop = ["utrykningstid", "responstid", "geometry"]
        df_incidents.drop(columns_to_drop, axis=1, inplace=True)
        # drop the few rows with errors in triage code
        df_incidents = df_incidents[df_incidents["hastegrad"] != "V"]
        # fill NaN values
        columns_to_fill = ["hastegrad", "varslet", "rykker_ut", "ank_hentested", "avg_hentested", "ank_levsted", "ledig"]
        for col in columns_to_fill:
            df_incidents[col].fillna("", inplace=True)
        # convert to correct time formats
        time_columns = ["tidspunkt", "tiltak_opprettet", "varslet", "rykker_ut", "ank_hentested", "avg_hentested", "ank_levsted", "ledig"]
        for col in time_columns:
            df_incidents[col] = df_incidents[col].apply(self._convert_time_format)

        df_incidents.to_csv(self._clean_incidents_data_path, index=False)

    def _convert_time_format(self, x: str) -> str:
        if x != "":
            return pd.to_datetime(x, format="%d.%m.%Y  %H:%M:%S ").strftime("%Y.%m.%dT%H:%M:%S")

    def _invalid_date_format(self, date_string: str) -> bool:
        """Helper function used for checking date formats are consistent."""
        pattern = regex.compile(r"^\d{2}\.\d{2}\.\d{4}  \d{2}:\d{2}:\d{2} $|^$")
        return not pattern.match(date_string)

    def _process_incidents(self) -> None:
        # load cleaned dataset
        df_incidents_clean = pd.read_csv(self._clean_incidents_data_path, low_memory=False)
        # create columns
        column_data_types = {
            "id": "int32",
            "synthetic": "bool",
            "triage_impression_during_call": "str",
            "time_call_received": "str",
            "time_incident_created": "str",
            "time_resource_appointed": "str",
            "time_ambulance_dispatch_to_scene": "str",
            "time_ambulance_arrived_at_scene": "str",
            "time_ambulance_dispatch_to_hospital": "str",
            "time_ambulance_arrived_at_hospital": "str",
            "time_ambulance_available": "str",
            "response_time_sec": "float32",
            "longitude": "float32",
            "latitude": "float32",
            "easting": "int32",
            "northing": "int32",
            "grid_id": "int64",
            "grid_row": "int32",
            "grid_col": "int32",
            "region": "str",
            "urban_settlement": "bool"
        }
        # create rows
        row_data = {
            "id": [],
            "synthetic": [],
            "triage_impression_during_call": [],
            "time_call_received": [],
            "time_incident_created": [],
            "time_resource_appointed": [],
            "time_ambulance_dispatch_to_scene": [],
            "time_ambulance_arrived_at_scene": [],
            "time_ambulance_dispatch_to_hospital": [],
            "time_ambulance_arrived_at_hospital": [],
            "time_ambulance_available": [],
            "response_time_sec": [],
            "longitude": [],
            "latitude": [],
            "easting": [],
            "northing": [],
            "grid_id": [],
            "grid_row": [],
            "grid_col": [],
            "region": [],
            "urban_settlement": []
        }

        gdf_oslo_bounds = utils.get_bounds(file_paths=[os.path.join(constants.PROJECT_DIRECTORY_PATH, "data", "ssb_2019_oslo_polygon_epsg4326.geojson")])
        gdf_akershus_bounds = utils.get_bounds(file_paths=[os.path.join(constants.PROJECT_DIRECTORY_PATH, "data", "ssb_2019_akershus_polygon_epsg4326.geojson")])
        gdf_urban_settlement_bounds = utils.get_bounds(file_paths=[os.path.join(constants.PROJECT_DIRECTORY_PATH, "data", "ssb_2021_urban_settlements_polygon_epsg4326.geojson")])

        grid_id_mapping = {}
        # rename columns
        for _, row in df_incidents_clean.iterrows():
            row_data["id"].append(row["id"])
            row_data["synthetic"].append(False)
            row_data["triage_impression_during_call"].append(row["hastegrad"])
            row_data["time_call_received"].append(row["tidspunkt"])
            row_data["time_incident_created"].append(row["tiltak_opprettet"])
            row_data["time_resource_appointed"].append(row["varslet"])
            row_data["time_ambulance_dispatch_to_scene"].append(row["rykker_ut"])
            row_data["time_ambulance_arrived_at_scene"].append(row["ank_hentested"])
            row_data["time_ambulance_dispatch_to_hospital"].append(row["avg_hentested"])
            row_data["time_ambulance_arrived_at_hospital"].append(row["ank_levsted"])
            row_data["time_ambulance_available"].append(row["ledig"])
            # get response time
            row["tidspunkt"] = pd.to_datetime(row["tidspunkt"], format="%Y.%m.%dT%H:%M:%S")
            row["ank_hentested"] = pd.to_datetime(row["ank_hentested"], format="%Y.%m.%dT%H:%M:%S")
            row_data["response_time_sec"].append((row["ank_hentested"] - row["tidspunkt"]).total_seconds())
            # get geo data
            easting, northing = row["xcoor"], row["ycoor"]
            lon, lat = utils.utm_to_geographic(easting, northing)
            grid_id = utils.utm_to_id(easting, northing)
            grid_row, grid_col = utils.id_to_row_col(grid_id)

            row_data["longitude"].append(lon)
            row_data["latitude"].append(lat)
            row_data["easting"].append(easting)
            row_data["northing"].append(northing)
            row_data["grid_id"].append(grid_id)
            row_data["grid_row"].append(grid_row)
            row_data["grid_col"].append(grid_col)

            row_data["region"].append(np.nan)
            row_data["urban_settlement"].append(False)
            for cell_corner in utils.get_cell_corners(easting, northing):
                grid_id_mapping[cell_corner] = (np.nan, False)
        # set region and urban_settlement
        for key in grid_id_mapping:
            lon, lat = key
            region, urban_settlement = grid_id_mapping[key]
            point = shapely.geometry.Point(lon, lat)
            if gdf_akershus_bounds.contains(point).any():
                region = "Akershus"
            elif gdf_oslo_bounds.contains(point).any():
                region = "Oslo"
            urban_settlement = gdf_urban_settlement_bounds.contains(point).any()
            grid_id_mapping[key] = (region, urban_settlement)

        for idx in range(len(row_data["region"])):
            cell_corners = utils.get_cell_corners(row_data["easting"][idx], row_data["northing"][idx])

            region = row_data["region"][idx]
            urban_settlement = row_data["urban_settlement"][idx]
            # get urban_settlement value
            for cell_corner in cell_corners:
                _, new_urban_settlement = grid_id_mapping[cell_corner]
                if new_urban_settlement:
                    urban_settlement = new_urban_settlement
                    break

            row_data["urban_settlement"][idx] = urban_settlement
            # get region value
            oslo_count = 0
            akershus_count = 0
            for cell_corner in cell_corners:
                new_region, _ = grid_id_mapping[cell_corner]
                if new_region == "Oslo":
                    oslo_count += 1
                elif new_region == "Akershus":
                    akershus_count += 1

            if (oslo_count + akershus_count) == 0:
                region = np.nan
            elif oslo_count >= akershus_count:
                region = "Oslo"
            else:
                region = "Akershus"

            row_data["region"][idx] = region
        # convert to dataframe
        df_incidents = pd.DataFrame(row_data)
        for column, dtype in column_data_types.items():
            df_incidents[column] = df_incidents[column].astype(dtype)
        # convert the triage codes
        df_incidents = self._rename_triage_categories(df_incidents)
        # sort
        df_incidents["time_call_received"] = pd.to_datetime(df_incidents["time_call_received"], format="%Y.%m.%dT%H:%M:%S")
        df_incidents.sort_values(by="time_call_received", inplace=True)
        df_incidents["time_call_received"] = df_incidents["time_call_received"].dt.strftime("%Y.%m.%dT%H:%M:%S")
        # save to disk
        df_incidents.to_csv(self._processed_incidents_data_path, index=False)

    def _process_depots(self) -> None:
        # load cleaned dataset
        df_depots_clean = pd.read_csv(self._clean_depots_data_path)
        # create columns
        column_data_types = {
            "type": "str",
            "static": "bool",
            "longitude": "float32",
            "latitude": "float32",
            "easting": "int32",
            "northing": "int32",
            "grid_id": "int64",
            "grid_row": "int32",
            "grid_col": "int32"
        }
        # create rows
        row_data = {
            "type": [],
            "static": [],
            "longitude": [],
            "latitude": [],
            "easting": [],
            "northing": [],
            "grid_id": [],
            "grid_row": [],
            "grid_col": []
        }
        # iterate over cleaned dataset
        for _, row in df_depots_clean.iterrows():
            type = None
            if row["type"] == "stasjon":
                type = "Depot"
            elif row["type"] == "beredskapspunkt":
                type = "Depot"
            elif row["type"] == "sykehus":
                type = "Hospital"
            row_data["type"].append(type)
            row_data["static"].append(False if row["type"] == "beredskapspunkt" else True)
            # get geo data
            easting, northing = row["easting"], row["northing"]
            lon, lat = utils.utm_to_geographic(easting, northing)
            grid_id = utils.utm_to_id(easting, northing)
            grid_row, grid_col = utils.id_to_row_col(grid_id)

            row_data["longitude"].append(lon)
            row_data["latitude"].append(lat)
            row_data["easting"].append(easting)
            row_data["northing"].append(northing)
            row_data["grid_id"].append(grid_id)
            row_data["grid_row"].append(grid_row)
            row_data["grid_col"].append(grid_col)
        # convert to dataframe and save to disk
        df_depots = pd.DataFrame(row_data)
        for column, dtype in column_data_types.items():
            df_depots[column] = df_depots[column].astype(dtype)
        df_depots.to_csv(self._processed_depots_data_path, index=False)

    def _rename_triage_categories(self, df: pd.DataFrame, column_name="triage_impression_during_call") -> pd.DataFrame:
        """
        Rename the triage categories in the specified column of the dataframe.

        Parameters:
        - df: pandas.DataFrame
        - column_name: str, the name of the column to be modified.

        Returns:
        - df: pandas.DataFrame with updated triage category names.
        """
        df[column_name] = df[column_name].map(constants.TRIAGE_MAPPING)

        return df

    def _enhance_incidents(self) -> None:
        # load processed dataset
        df_incidents = pd.read_csv(self._processed_incidents_data_path, low_memory=False)
        # drop rows with NaN values
        df_incidents.dropna(subset=["time_ambulance_available", "time_ambulance_dispatch_to_scene", "triage_impression_during_call", "time_resource_appointed", "region"], inplace=True)
        # drop rows where time_ambulance_arrived_at_scene or time_ambulance_dispatch_to_hospital does not exist, but time_ambulance_arrived_at_hospital exists
        mask1 = df_incidents["time_ambulance_arrived_at_scene"].isna() & df_incidents["time_ambulance_arrived_at_hospital"].notna()
        mask2 = df_incidents["time_ambulance_dispatch_to_hospital"].isna() & df_incidents["time_ambulance_arrived_at_hospital"].notna()
        mask3 = df_incidents["time_ambulance_dispatch_to_hospital"].notna() & df_incidents["time_ambulance_arrived_at_hospital"].isna()
        df_incidents = df_incidents[~(mask1 | mask2 | mask3)]
        # drop rows with 'Moderate Priority' or 'Scheduled'
        df_incidents = df_incidents.query('triage_impression_during_call not in ["V1", "V2"]').copy()
        # fix rows with negative time frames
        df_incidents = fix_timeframes(df_incidents)
        # remove outliers
        df_incidents = remove_outliers_pdf(df_incidents, 'response_time_sec')
        # sort
        df_incidents["time_call_received"] = pd.to_datetime(df_incidents["time_call_received"], format="%Y.%m.%dT%H:%M:%S")
        df_incidents.sort_values(by="time_call_received", inplace=True)
        df_incidents["time_call_received"] = df_incidents["time_call_received"].dt.strftime("%Y.%m.%dT%H:%M:%S")
        # save to disk
        df_incidents.to_csv(self._enhanced_incidents_data_path, index=False)

    def _enhance_depots(self) -> None:
        df_depots = pd.read_csv(self._processed_depots_data_path)
        df_depots.to_csv(self._enhanced_depots_data_path, index=False)


class DataLoader:
    """Class for loading dataset."""

    def __init__(self, data_preprocessor: DataPreprocessor = DataPreprocessorOUS) -> None:
        """Initialize the data loader."""
        self.data_preprocessor: DataPreprocessor = data_preprocessor()

        self.clean_incidents_df: pd.DataFrame = None
        self.clean_depots_df: pd.DataFrame = None
        self.processed_incidents_df: pd.DataFrame = None
        self.processed_depots_df: pd.DataFrame = None
        self.enhanced_incidents_df: pd.DataFrame = None
        self.enhanced_depots_df: pd.DataFrame = None

    def execute(self, clean=True, processed=True, enhanced=True) -> None:
        """Run the data loader."""
        if not clean and not processed and not enhanced:
            return

        if clean and (not os.path.exists(self.data_preprocessor._clean_incidents_data_path) or not os.path.exists(self.data_preprocessor._clean_depots_data_path)):
            raise Exception("Missing the cleaned data files.")
        if processed and (not os.path.exists(self.data_preprocessor._processed_incidents_data_path) or not os.path.exists(self.data_preprocessor._processed_depots_data_path)):
            raise Exception("Missing the processed data files.")
        if enhanced and (not os.path.exists(self.data_preprocessor._enhanced_incidents_data_path) or not os.path.exists(self.data_preprocessor._enhanced_depots_data_path)):
            raise Exception("Missing the enhanced data files.")

        progress_bar = tqdm(desc="Loading dataset", total=(clean + processed + enhanced) * 2)

        if clean:
            self.cleaned_incidents_df = self.data_preprocessor.load_clean_incidents_dataframe()
            progress_bar.update(1)
            self.cleaned_depots_df = self.data_preprocessor.load_clean_depots_dataframe()
            progress_bar.update(1)

        if processed:
            self.processed_incidents_df = self.data_preprocessor.load_processed_incidents_dataframe()
            progress_bar.update(1)
            self.processed_depots_df = self.data_preprocessor.load_processed_depots_dataframe()
            progress_bar.update(1)

        if enhanced:
            self.enhanced_incidents_df = self.data_preprocessor.load_enhanced_incidents_dataframe()
            progress_bar.update(1)
            self.enhanced_depots_df = self.data_preprocessor.load_enhanced_depots_dataframe()
            progress_bar.update(1)


def fix_timeframes(df_incidents: pd.DataFrame) -> pd.DataFrame:
    # convert time columns to datetime format
    time_columns = [
        'time_call_received', 'time_incident_created', 'time_resource_appointed',
        'time_ambulance_dispatch_to_scene', 'time_ambulance_arrived_at_scene', 'time_ambulance_dispatch_to_hospital',
        'time_ambulance_arrived_at_hospital', 'time_ambulance_available'
    ]
    df_incidents[time_columns] = df_incidents[time_columns].apply(pd.to_datetime, errors='coerce', format="%Y.%m.%dT%H:%M:%S")

    # iterate over each pair of time columns in order
    for i in range(len(time_columns) - 1):
        col1, col2 = time_columns[i], time_columns[i + 1]

        # find rows where the later time is before the earlier time
        incorrect_order_mask = df_incidents[col2] < df_incidents[col1]

        # calculate median time difference based on all other correct rows
        correct_order_mask = df_incidents[col2] >= df_incidents[col1]
        median_time_diff = (df_incidents.loc[correct_order_mask, col2] - df_incidents.loc[correct_order_mask, col1]).dt.total_seconds().median()

        # correct the later timestamp by setting it to the earlier timestamp plus the median time difference
        num_incorrect = incorrect_order_mask.sum()
        if num_incorrect > 0:
            df_incidents.loc[incorrect_order_mask, col2] = df_incidents.loc[incorrect_order_mask, col1] + pd.Timedelta(seconds=median_time_diff)
            df_incidents.loc[incorrect_order_mask, 'synthetic'] = True
    # recalculate response time
    df_incidents['response_time_sec'] = (df_incidents['time_ambulance_arrived_at_scene'] - df_incidents['time_call_received']).dt.total_seconds()

    # convert time columns back to string format if needed
    df_incidents[time_columns] = df_incidents[time_columns].map(lambda x: x.strftime('%Y.%m.%dT%H:%M:%S') if not pd.isnull(x) else '')

    return df_incidents


def remove_outliers_pdf(df, column_name, threshold=0.01):
    # calculate the PDF values of the log-transformed column
    df_log = np.log1p(df[column_name])
    mean_log = np.mean(df_log)
    std_log = np.std(df_log)
    pdf_values = norm.pdf(df_log, mean_log, std_log)

    # identify the rows where the PDF value is above the threshold
    mask = pdf_values >= threshold

    return df[mask]
