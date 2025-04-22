## main.py calles other analysis scripts, serving as a wrapper function
from prep-acoustic-data import prep_acoustic_data
from extract-feature import extract_feature
from concat-features import concat

import os
dataPath = "C:/Users/86139/Desktop/FishTetherExperiment/ProcessedData"
inputPath = os.path.join(dataPath, "processed_AllFishCombined.csv")
outputPath = os.path.join(dataPath, "acousticDataFrame.csv")

# Prepare accoustic data for analysis
acousticData = prep_acoustic_data(inputPath, outputPath)

# For each fish, extract PCs and scores from them
fishList = ["LT001", "BUR001", "LWF001", "SMB001"]
featureExtracted = concat_features(fishList, acousticData, outputPath)
Z3 = featureExtracted["Z3"]

# Clustering

# Visualize Clustering Results in 3D space

