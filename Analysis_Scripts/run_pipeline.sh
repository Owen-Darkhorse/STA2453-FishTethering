#!/bin/bash

# Run the R script
echo "Running R script to concatenate and tidy EV files"

if [-f "ProcessData/processed_AllFishCombined_unfiltered.csv"]
  echo "Model data already exists, skipping R script."
  exit 0
else
  echo "Model data does not exist, running R script."
  Rscript Analysis_Scripts/R-script/concatenate_tidy_EVfiles.R
fi

# Check if the R script ran successfully
if [ $? -ne 0 ]; then
  echo "R script failed. Exiting."
  exit 1
fi

# Run the Python script
echo "Running Python script..."
python3 Analysis_Scripts/model-scripts/__main__.py

# Check if the Python script ran successfully
if [ $? -ne 0 ]; then
  echo "Python script failed. Exiting."
  exit 1
fi

echo "All scripts ran successfully!"