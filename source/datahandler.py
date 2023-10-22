import constants

import os
import typing


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
        pass

    def _clean_data(self) -> typing.Any:
        """Clean the raw data and cache it."""
        pass

    def _process_data(self) -> typing.Any:
        """Process the clean data and cache it."""
        pass
