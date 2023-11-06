import datahandler
import pathing


def main():
    """Main program."""
    data_preprocessor = datahandler.DataPreprocessorOUS(dataset_id="oslo")
    data_preprocessor.execute()

    od = pathing.OriginDestination(
        dataset_id="oslo",
        graph_central_location=(59.978023, 11.041620),
        grap_distance=70000,
        utm_epsg=f"EPSG:326{33}"
    )
    od.build()


if __name__ == "__main__":
    main()
