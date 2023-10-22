import constants

import os
import pandas as pd
import regex
from tqdm import tqdm


class DataPreprocessor:
    """Abstract class for preprocessing datasets."""

    def __init__(self, dataset_id: str):
        """
        Initialize the data preprocessor.

        Args:
            dataset_id: Unique identifier for the dataset.
        """
        self.dataset_id = dataset_id
        # paths for raw data
        raw_directory_path = os.path.join(constants.PROJECT_DIRECTORY_PATH, "data", "raw", self.dataset_id)
        self._raw_incidents_data_path = os.path.join(raw_directory_path, "incidents.csv")
        self._raw_depots_data_path = os.path.join(raw_directory_path, "depots.csv")
        # paths for cleaned data
        clean_directory_path = os.path.join(constants.PROJECT_DIRECTORY_PATH, "data", "clean", self.dataset_id)
        self._clean_incidents_data_path = os.path.join(clean_directory_path, "incidents.csv")
        self._clean_depots_data_path = os.path.join(clean_directory_path, "depots.csv")
        # paths for processed data
        processed_directory_path = os.path.join(constants.PROJECT_DIRECTORY_PATH, "data", "processed", self.dataset_id)
        self._processed_incidents_data_path = os.path.join(processed_directory_path, "incidents.csv")
        self._processed_depots_data_path = os.path.join(processed_directory_path, "depots.csv")

    def execute(self) -> None:
        """Run the preprocessor to clean and process the dataset."""
        self._clean_data()

    def _clean_data(self) -> None:
        """Clean the raw data and cache it."""
        os.makedirs(os.path.dirname(self._clean_incidents_data_path), exist_ok=True)
        pass

    def _process_data(self) -> None:
        """Process the clean data and cache it."""
        os.makedirs(os.path.dirname(self._processed_incidents_data_path), exist_ok=True)
        pass


class DataPreprocessorOUS(DataPreprocessor):
    """Class for preprocessing the OUS dataset."""

    def __init__(self, dataset_id: str):
        super().__init__(dataset_id)

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

    def _fix_csv(self):
        with open(self._raw_incidents_data_path, "r", encoding="windows-1252") as input_file, \
             open(self._clean_incidents_data_path, "w", encoding="utf-8") as output_file:

            header = input_file.readline().replace('""', '"id"')
            output_file.write(header)

            for line in input_file:
                if regex.match(r".*\(.*,.*\).*", line):
                    line = regex.sub(r"\([^,()]+\K,", "\\,", line)
                output_file.write(line)

    def _clean_and_save_incidents(self):
        df_incidents = pd.read_csv(self._clean_incidents_data_path, usecols=range(32), escapechar="\\", low_memory=False)
        # drop unnecessary columns
        columns_to_drop = ["utrykningstid", "responstid"]
        df_incidents.drop(columns_to_drop, axis=1, inplace=True)
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
            return pd.to_datetime(x, format="%d.%m.%Y  %H:%M:%S ").strftime("%d.%m.%YT%H:%M:%S")
        return ""


    def _process_data(self) -> None:
        super()._process_data()


def invalid_date_format(date_string):
    """Helper function used for checking date formats are consistent."""
    # Pattern for the date format "13.02.2015 09:37:14 " and ""
    pattern = regex.compile(r"^\d{2}\.\d{2}\.\d{4}  \d{2}:\d{2}:\d{2} $|^$")
    return not pattern.match(date_string)
