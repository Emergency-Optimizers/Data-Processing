import datahandler
import pathing


def main():
    """Main program."""
    #data_preprocessor = datahandler.DataPreprocessorOUS(dataset_id="oslo")
    #data_preprocessor.execute()

    od = pathing.OriginDestination(
        dataset_id="oslo",
        utm_epsg=f"EPSG:326{33}"
    )
    od.build()


if __name__ == "__main__":
    main()
