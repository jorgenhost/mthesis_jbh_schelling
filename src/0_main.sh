#!/bin/bash


# Make conda env (see env.yaml)
# Specify conda env path
CONDA_PATH=C:/ProgramData/Anaconda3
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate mthesis_main

# Base dir
BASE_DIR = "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Bash: Fetch admin data, optimize data types"
python src/1_data_parse.py

echo "Bash: Make dataset with address id and person identifiers"
python src/2.1_merge.py

echo "Bash: Map people & sequences to household..."
python src/2.2_network_householdz.py

echo "Bash: Link neighborhood to address id..."
python src/2.3_neighborhood_maxp.py

echo "Bash: Building KNN network of households..."
python src/2.4_KNN_network.py

echo "Bash: Building KNN network of households, large K..."
python src/2.4.2_KNN_network_big.py

echo "Bash: KNN descriptives"
python src/2.5_KNN_descriptives.py

echo "Bash: KNN descriptives (maps)"
python src/2.6_KNN_descriptives_maps.py

echo "Bash: Make KNN panel"
python src/2.7_KNN_panel.py

echo "Bash: Run all regressions"
python src/3_regz.py