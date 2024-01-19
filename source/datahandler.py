import constants
import utils

import os
import pandas as pd
import numpy as np
import regex
from tqdm import tqdm
from scipy.stats import norm
import geopandas as gpd
import shapely.geometry


class DataLoader:
    """Class for loading dataset."""

    def __init__(self, dataset_id: str) -> None:
        """
        Initialize the data loader.

        Args:
            dataset_id: Unique identifier for the dataset.
        """
        self.dataset_id = dataset_id
        self.cleaned_incidents_df: pd.DataFrame = None
        self.cleaned_depots_df: pd.DataFrame = None
        self.processed_incidents_df: pd.DataFrame = None
        self.processed_depots_df: pd.DataFrame = None
        self.enhanced_incidents_df: pd.DataFrame = None
        self.enhanced_depots_df: pd.DataFrame = None
        # paths for cleaned data
        self._clean_incidents_data_path = utils.get_clean_incidents_path(self.dataset_id)
        self._clean_depots_data_path = utils.get_clean_depots_path(self.dataset_id)
        # paths for processed data
        self._processed_incidents_data_path = utils.get_processed_incidents_path(self.dataset_id)
        self._processed_depots_data_path = utils.get_processed_depots_path(self.dataset_id)
        # paths for enhanced data
        self._enhanced_incidents_data_path = utils.get_enhanced_incidents_path(self.dataset_id)
        self._enhanced_depots_data_path = utils.get_enhanced_depots_path(self.dataset_id)

    def execute(self, clean = True, processed = True, enhanced = True) -> None:
        """Run the data loader."""
        if not clean and not processed and not enhanced:
            return

        if clean and not os.path.exists(self._clean_incidents_data_path) or not os.path.exists(self._clean_depots_data_path):
            raise Exception("Missing the cleaned data files.")
        if processed and not os.path.exists(self._processed_incidents_data_path) or not os.path.exists(self._processed_depots_data_path):
            raise Exception("Missing the processed data files.")
        if enhanced and not os.path.exists(self._enhanced_incidents_data_path) or not os.path.exists(self._enhanced_depots_data_path):
            raise Exception("Missing the enhanced data files.")

        progress_bar = tqdm(desc="Loading dataset", total=(clean + processed + enhanced))

        if clean:
            self.cleaned_incidents_df = pd.read_csv(self._clean_incidents_data_path, low_memory=False)
            progress_bar.update(1)
            self.cleaned_depots_df = pd.read_csv(self._clean_depots_data_path, low_memory=False)
            progress_bar.update(1)

        if processed:
            self.processed_incidents_df = pd.read_csv(self._processed_incidents_data_path, low_memory=False)
            progress_bar.update(1)
            self.processed_depots_df = pd.read_csv(self._processed_depots_data_path, low_memory=False)
            progress_bar.update(1)

        if enhanced:
            self.enhanced_incidents_df = pd.read_csv(self._enhanced_incidents_data_path, low_memory=False)
            progress_bar.update(1)
            self.enhanced_depots_df = pd.read_csv(self._enhanced_depots_data_path, low_memory=False)
            progress_bar.update(1)


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


class DataPreprocessorOUS_V2(DataPreprocessor):
    """Class for preprocessing the OUS dataset. Version 2.0"""

    def __init__(self) -> None:
        super().__init__(dataset_id="oslo")
    
    def _clean_incidents(self) -> None:
        super()._clean_incidents()

        self.fix_csv_errors()
        dataframe = pd.read_csv(self._clean_incidents_data_path, escapechar="\\", low_memory=False)
        dataframe = self.split_geometry(dataframe)
        dataframe = self.drop_unnecessary_raw_columns(dataframe)
        dataframe = self.fix_raw_types(dataframe)

        self.save_dataframe(dataframe, self._clean_incidents_data_path)
    
    def fix_csv_errors(self) -> None:
        """Fixes common errors in a CSV file and saves the corrected version."""
        with open(self._raw_incidents_data_path, "r", encoding="windows-1252") as source_file, \
            open(self._clean_incidents_data_path, "w", encoding="utf-8") as target_file:
            
            # fix empty header
            header = source_file.readline().replace('""', '"id"')
            target_file.write(header)
            
            # fix comma errors in the data lines
            for line in source_file:
                if regex.match(r".*\(.*,.*\).*", line):
                    line = regex.sub(r"\([^,()]+\K,", "\\,", line)

                target_file.write(line)
    
    def split_geometry(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Splits the 'geometry' column of the DataFrame into two separate columns.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the 'geometry' column.

        Returns:
            pd.DataFrame: The DataFrame with the 'geometry' column split into 'geometry_x' and 'geometry_y'.
        """
        # splitting the 'geometry' column into two new columns
        geometry_split = dataframe["geometry"].str.replace("c\(|\)", "", regex=True).str.split(", ", expand=True)
        dataframe[["geometry_x", "geometry_y"]] = geometry_split

        # drop the problematic column
        dataframe.drop(["geometry"], axis=1, inplace=True)

        return dataframe

    def drop_unnecessary_raw_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        columns_to_drop = [
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
            "statistikkÅr"
        ]
        dataframe.drop(columns_to_drop, axis=1, inplace=True)

        return dataframe

    def fix_raw_types(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        headers_types = {
            "id": "int64",
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
            "geometry_x": "int64",
            "geometry_y": "int64"
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

        for col in date_columns:
            dataframe[col] = pd.to_datetime(dataframe[col], format="%d.%m.%Y %H:%M:%S ", errors="coerce")

        return dataframe
    
    def _clean_depots(self) -> None:
        super()._clean_depots()

        dataframe = pd.read_csv(self._raw_depots_data_path)
        self.save_dataframe(dataframe, self._clean_depots_data_path)
    
    def _process_incidents(self) -> None:
        super()._process_incidents()

        dataframe = self.initialize_processed_incidents_dataframe()
        dataframe = self.add_geo_data(dataframe)

        self.save_dataframe(dataframe, self._processed_incidents_data_path)
    
    def initialize_processed_incidents_dataframe(self) -> pd.DataFrame:
        dataframe_clean = self.load_clean_incident_dataframe()

        dataframe = pd.DataFrame()
        dataframe["id"] = dataframe_clean["id"]
        dataframe["triage_impression_during_call"] = dataframe_clean["hastegrad"]
        dataframe["time_call_received"] = dataframe_clean["tidspunkt"]
        dataframe["time_call_processed"] = dataframe_clean["tiltak_opprettet"]
        dataframe["time_ambulance_notified"] = dataframe_clean["varslet"]
        dataframe["time_dispatch"] = dataframe_clean["rykker_ut"]
        dataframe["time_arrival_scene"] = dataframe_clean["ank_hentested"]
        dataframe["time_departure_scene"] = dataframe_clean["avg_hentested"]
        dataframe["time_arrival_hospital"] = dataframe_clean["ank_levsted"]
        dataframe["time_available"] = dataframe_clean["ledig"]
        dataframe["grid_id"] = dataframe_clean["ssbid1000M"]
        dataframe["x"] = dataframe_clean["xcoor"]
        dataframe["y"] = dataframe_clean["ycoor"]
        dataframe["x_accurate"] = dataframe_clean["geometry_x"]
        dataframe["y_accurate"] = dataframe_clean["geometry_y"]
        dataframe["longitude"] = np.nan
        dataframe["latitude"] = np.nan
        dataframe["region"] = None
        dataframe["urban_settlement"] = False

        return dataframe
    
    def _process_depots(self) -> None:
        super()._process_depots()

        dataframe = self.initialize_processed_depots_dataframe()
        dataframe = self.convert_depot_types(dataframe)
        dataframe = self.add_grid_id(dataframe)
        dataframe = self.add_geo_data(dataframe)

        self.save_dataframe(dataframe, self._processed_depots_data_path)
    
    def initialize_processed_depots_dataframe(self) -> pd.DataFrame:
        dataframe_clean = self.load_clean_depots_dataframe()

        dataframe = pd.DataFrame()
        dataframe["type"] = dataframe_clean["type"]
        dataframe["grid_id"] = 0
        dataframe["x"] = 0
        dataframe["y"] = 0
        dataframe["x_accurate"] = dataframe_clean["easting"]
        dataframe["y_accurate"] = dataframe_clean["northing"]
        dataframe["longitude"] = np.nan
        dataframe["latitude"] = np.nan
        dataframe["region"] = None
        dataframe["urban_settlement"] = False

        return dataframe
    
    def convert_depot_types(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        for index, _ in dataframe.iterrows():
            depot_type = dataframe.at[index, "type"]

            new_depot_type = None
            if depot_type == "stasjon":
                new_depot_type = "Depot"
            elif depot_type == "beredskapspunkt":
                new_depot_type = "Depot"
            elif depot_type == "sykehus":
                new_depot_type = "Hospital"
            
            dataframe.at[index, "type"] = new_depot_type
        
        return dataframe
    
    def add_grid_id(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        for index, _ in dataframe.iterrows():
            x_accurate = dataframe.at[index, "x_accurate"]
            y_accurate = dataframe.at[index, "y_accurate"]

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

        cached_geo_data_point = {}
        cached_geo_data_grid_id = {}

        for index, _ in dataframe.iterrows():
            grid_id = dataframe.at[index, "grid_id"]

            if grid_id in cached_geo_data_grid_id:
                longitude, latitude, region, urban_settlement = cached_geo_data_grid_id[grid_id]
            else:
                x = dataframe.at[index, "x"]
                y = dataframe.at[index, "y"]

                region_akershus_count = 0
                region_oslo_count = 0
                urban_settlement_count = 0

                for longitude, latitude in utils.get_cell_corners(x, y):
                    if (longitude, latitude) in cached_geo_data_point:
                        region, urban_settlement = cached_geo_data_point[(longitude, latitude)]
                    else:
                        point = shapely.geometry.Point(longitude, latitude)

                        region = None
                        if gdf_akershus_bounds.contains(point).any():
                            region = "Akershus"
                        elif gdf_oslo_bounds.contains(point).any():
                            region = "Oslo"

                        urban_settlement = gdf_urban_settlement_bounds.contains(point).any()

                        cached_geo_data_point[(longitude, latitude)] = (region, urban_settlement)

                    if region == "Akershus":
                        region_akershus_count += 1
                    elif region == "Oslo":
                        region_oslo_count += 1
                        
                    urban_settlement_count += urban_settlement
                
                if (region_akershus_count + region_oslo_count) == 0:
                    region = None
                elif region_oslo_count >= region_akershus_count:
                    region = "Oslo"
                else:
                    region = "Akershus"
                
                urban_settlement = urban_settlement_count != 0

                cached_geo_data_grid_id[grid_id] = (longitude, latitude, region, urban_settlement)
            
            dataframe.at[index, "longitude"] = longitude
            dataframe.at[index, "latitude"] = latitude
            dataframe.at[index, "region"] = region
            dataframe.at[index, "urban_settlement"] = urban_settlement
        
        return dataframe
    
    def _enhance_incidents(self) -> None:
        super()._enhance_incidents()
    
    def _enhance_depots(self) -> None:
        super()._enhance_depots()
    
    def save_dataframe(self, dataframe: pd.DataFrame, filepath: str):
        dataframe.to_csv(filepath, index=False)
    
    def load_clean_incident_dataframe(self) -> pd.DataFrame:
        column_types = {
            "id": "int64",
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
        column_index_dates = [2, 3, 6, 7, 8, 9, 10, 11]

        dataframe = pd.read_csv(
            self._clean_incidents_data_path,
            dtype=column_types,
            na_values=[""],
            parse_dates=column_index_dates
        )

        return dataframe

    def load_clean_depots_dataframe(self) -> pd.DataFrame:
        column_types = {
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
    
    def load_processed_incident_dataframe(self) -> pd.DataFrame:
        column_types = {
            "id": "int64",
            "triage_impression_during_call": "object",
            "grid_id": "int64",
            "x": "int64",
            "y": "int64",
            "x_accurate": "int64",
            "y_accurate": "int64",
            "longitude": "float64",
            "latitude": "float64",
            "region": "object",
            "urban_settlement": "bool"
        }
        column_index_dates = [2, 3, 4, 5, 6, 7, 8, 9]

        dataframe = pd.read_csv(
            self._processed_incidents_data_path,
            dtype=column_types,
            na_values=[""],
            parse_dates=column_index_dates
        )

        return dataframe
    
    def load_processed_depots_dataframe(self) -> pd.DataFrame:
        column_types = {
            "type": "object",
            "grid_id": "int64",
            "x": "int64",
            "y": "int64",
            "x_accurate": "int64",
            "y_accurate": "int64",
            "longitude": "float64",
            "latitude": "float64",
            "region": "object",
            "urban_settlement": "bool"
        }

        dataframe = pd.read_csv(
            self._processed_depots_data_path,
            dtype=column_types,
            na_values=[""]
        )

        return dataframe


class DataPreprocessorOUS(DataPreprocessor):
    """Class for preprocessing the OUS dataset."""

    def __init__(self) -> None:
        super().__init__(dataset_id="oslo")

    def _clean_incidents(self) -> None:
        super()._clean_incidents()

        self._fix_csv()
        self._clean_and_save_incidents()

    def _clean_depots(self) -> None:
        super()._clean_depots()

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
        # Pattern for the date format "13.02.2015 09:37:14 " and ""
        pattern = regex.compile(r"^\d{2}\.\d{2}\.\d{4}  \d{2}:\d{2}:\d{2} $|^$")
        return not pattern.match(date_string)

    def _process_incidents(self) -> None:
        super()._process_incidents()

        # load cleaned dataset
        df_incidents_clean = pd.read_csv(self._clean_incidents_data_path, low_memory=False)
        # create columns
        column_data_types = {
            "id": "int32",
            "synthetic": "bool",
            "triage_impression_during_call": "str",
            "time_call_received": "str",
            "time_call_processed": "str",
            "time_ambulance_notified": "str",
            "time_dispatch": "str",
            "time_arrival_scene": "str",
            "time_departure_scene": "str",
            "time_arrival_hospital": "str",
            "time_available": "str",
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
            "time_call_processed": [],
            "time_ambulance_notified": [],
            "time_dispatch": [],
            "time_arrival_scene": [],
            "time_departure_scene": [],
            "time_arrival_hospital": [],
            "time_available": [],
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
            row_data["time_call_processed"].append(row["tiltak_opprettet"])
            row_data["time_ambulance_notified"].append(row["varslet"])
            row_data["time_dispatch"].append(row["rykker_ut"])
            row_data["time_arrival_scene"].append(row["ank_hentested"])
            row_data["time_departure_scene"].append(row["avg_hentested"])
            row_data["time_arrival_hospital"].append(row["ank_levsted"])
            row_data["time_available"].append(row["ledig"])
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
        super()._process_depots()

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
        super()._enhance_incidents()

        # load processed dataset
        df_incidents = pd.read_csv(self._processed_incidents_data_path, low_memory=False)
        # drop rows with NaN values
        df_incidents.dropna(subset=["time_available", "time_dispatch", "triage_impression_during_call", "time_ambulance_notified", "region"], inplace=True)
        # drop rows where time_arrival_scene or time_departure_scene does not exist, but time_arrival_hospital exists
        mask1 = df_incidents["time_arrival_scene"].isna() & df_incidents["time_arrival_hospital"].notna()
        mask2 = df_incidents["time_departure_scene"].isna() & df_incidents["time_arrival_hospital"].notna()
        mask3 = df_incidents["time_departure_scene"].notna() & df_incidents["time_arrival_hospital"].isna()
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
        super()._enhance_depots()

        df_depots = pd.read_csv(self._processed_depots_data_path)
        df_depots.to_csv(self._enhanced_depots_data_path, index=False)


def fix_timeframes(df_incidents: pd.DataFrame) -> pd.DataFrame:
    # convert time columns to datetime format
    time_columns = [
        'time_call_received', 'time_call_processed', 'time_ambulance_notified',
        'time_dispatch', 'time_arrival_scene', 'time_departure_scene',
        'time_arrival_hospital', 'time_available'
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
    df_incidents['response_time_sec'] = (df_incidents['time_arrival_scene'] - df_incidents['time_call_received']).dt.total_seconds()

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
