# Data-Processing
This is a repository for pre-processing and analyzing OUH incident dataset, aiming for an integrated solution for our simulation program. The simulator program is found [here](https://github.com/Emergency-Optimizers/Simulator).

## Setup
0. Install required python version **3.11**
1. Install required packages `pip install -r source/requirements.txt` (We recommend using virtual environment, follow guide under **Virtual Environment Setup** below and skip this step)
3. Change directory `cd source`
4. Run program `python main.py` to start the dataset pipeline

### Virtual Environment Setup
#### Windows
0. Get the package `pip install virtualenv`
1. Create a new empty instance of python 3.11 environment `py -3.11 -m venv ./.venv`
2. Activate the environment `source .venv/Scripts/activate`
3. Install the packages required by this project `pip install -r source/requirements.txt`

#### Linux
0. Get the package `pip install virtualenv`
1. Create a new empty instance of python 3.11 environment `python -m venv ./.venv`
2. Activate the environment `source .venv/bin/activate`
3. Install the packages required by this project `pip install -r source/requirements.txt`

## Structure
All the code is in the `source/` directory, with the analysis notebooks contained in the `source/analysis/` directory. Analysis of the experiment results is done in `source/analysis/simulator/`.

### Data
The data used in this project can't be made public and will therefore not be contained in this public repository.

The data is stored in the `data/` directory and contains 4 directories for the dataset processing pipeline; `raw/` which contains the raw versions of `incidents.csv` and `depots.csv`, `clean/` which containes the cleaned versions of the files (removing CSV errors), `processed/` which contains the the processed versions of the files (convert to our dataframe template), and lastly `enhanced/` which contains the final form of the OUH dataset which the simulator will use. The OD cost matrix and traffic data is included.

```
data/
    enhanced/
        oslo/
            depots.csv
            incidents.csv
    processed/
        oslo/
            depots.csv
            incidents.csv
    clean/
        oslo/
            depots.csv
            incidents.csv
    raw/
        oslo/
            depots.csv
            incidents.csv
    oslo/
        od_matrix.txt
        traffic.csv
```
