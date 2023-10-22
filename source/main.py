import datahandler


def main():
    """Main program."""
    data_preprocessor = datahandler.DataPreprocessorOUS(dataset_id="oslo")
    data_preprocessor.execute()


if __name__ == "__main__":
    main()
