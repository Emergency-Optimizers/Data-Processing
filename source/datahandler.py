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

    def execute(self) -> None:
        """Run the data loader."""
        if not os.path.exists(self._clean_incidents_data_path) or not os.path.exists(self._clean_depots_data_path):
            raise Exception("Missing the cleaned data files.")
        if not os.path.exists(self._processed_incidents_data_path) or not os.path.exists(self._processed_depots_data_path):
            raise Exception("Missing the processed data files.")
        if not os.path.exists(self._enhanced_incidents_data_path) or not os.path.exists(self._enhanced_depots_data_path):
            raise Exception("Missing the enhanced data files.")

        progress_bar = tqdm(desc="Loading dataset", total=6)

        self.cleaned_incidents_df = pd.read_csv(self._clean_incidents_data_path, low_memory=False)
        progress_bar.update(1)
        self.cleaned_depots_df = pd.read_csv(self._clean_depots_data_path, low_memory=False)
        progress_bar.update(1)

        self.processed_incidents_df = pd.read_csv(self._processed_incidents_data_path, low_memory=False)
        progress_bar.update(1)
        self.processed_depots_df = pd.read_csv(self._processed_depots_data_path, low_memory=False)
        progress_bar.update(1)

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

    def _process_data(self) -> None:
        """Process the clean data and cache it."""
        os.makedirs(os.path.dirname(self._processed_incidents_data_path), exist_ok=True)

    def _enhance_data(self) -> None:
        """Process the clean data and cache it."""
        os.makedirs(os.path.dirname(self._enhanced_incidents_data_path), exist_ok=True)


class DataPreprocessorOUS(DataPreprocessor):
    """Class for preprocessing the OUS dataset."""

    def _clean_data(self) -> None:
        super()._clean_data()
        progress_bar = tqdm(desc="Cleaning dataset", total=2)

        if not os.path.exists(self._clean_incidents_data_path):
            self._fix_csv()
            self._clean_and_save_incidents()
        progress_bar.update(1)

        if not os.path.exists(self._clean_depots_data_path):
            df_depots = pd.read_csv(self._raw_depots_data_path)
            df_depots.to_csv(self._clean_depots_data_path, index=False)
        progress_bar.update(1)

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

    def _process_data(self) -> None:
        super()._process_data()
        progress_bar = tqdm(desc="Processing dataset", total=2)

        if not os.path.exists(self._processed_incidents_data_path):
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
        progress_bar.update(1)

        if not os.path.exists(self._processed_depots_data_path):
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
        progress_bar.update(1)

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

    def _enhance_data(self) -> None:
        super()._enhance_data()
        progress_bar = tqdm(desc="Enhancing dataset", total=2)

        if not os.path.exists(self._enhanced_incidents_data_path):
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
        progress_bar.update(1)

        if not os.path.exists(self._enhanced_depots_data_path):
            df_depots = pd.read_csv(self._processed_depots_data_path)
            df_depots.to_csv(self._enhanced_depots_data_path, index=False)
        progress_bar.update(1)


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
