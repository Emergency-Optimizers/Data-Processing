# Data-Processing
This is a repository for pre-processing and analyzing ambulance datasets, aiming for an integrated solution for our simulation program.

## Setup
0. Install required python version **3.11**
1. Install required packages `pip install -r source/requirements.txt` (We recommend using virtual environment, follow guide under **Virtual Environment Setup** below and skip this step)
3. Change directory `cd source`
4. Run program `python main.py`

### Virtual Environment Setup
#### Windows
0. Get the package `pip install virtualenv`
1. Create a new empty instance of python 3.11 environment `py -3.11 -m venv ./.venv`
2. Activate the environment `.venv/Scripts/activate`
3. Install the packages required by this project `pip install -r source/requirements.txt`

#### Linux
0. Get the package `pip install virtualenv`
1. Create a new empty instance of python 3.11 environment `python -m venv ./.venv`
2. Activate the environment `source .venv/bin/activate`
3. Install the packages required by this project `pip install -r source/requirements.txt`

## Structure
### Data
The data used in this project can't be made public and will therefor not be contained in this public repository.
The data is stored in the `data/` directory and contains two directories; `raw/` which contains the location directory of `incidents.csv` and `depots.csv`, and `processed/` which contains the location directory of preprocessed `incidents.csv` and `depots.csv`.
Example:
```
data/
    processed/
        oslo/
            depots.csv
            incidents.csv
    raw/
        oslo/
            depots.csv
            incidents.csv
```
