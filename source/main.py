import datahandler
import pathing


def main():
    """Main program."""
    # runs data processing pipeline
    data_preprocessor = datahandler.DataPreprocessorOUS_V2()
    data_preprocessor.execute()

    # builds OD cost matrix
    od = pathing.OriginDestination(
        dataset_id="oslo",
        utm_epsg=f"EPSG:326{33}"
    )
    od.build()


if __name__ == "__main__":
    main()
