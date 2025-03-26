## main.py calles other analysis scripts, serving as a wrapper function
from prepAcousticData import prepAcousticData
from concatFeatures import concatFeatures
from plot3D import plot3DPoints
# from clustering import clusterer
# from visualizeClusters import plotResults

import os
dataPath = "C:\\Users\\86139\\Desktop\\FishTetherExperiment\\ProcessedData"
# inputPath = os.path.join(dataPath, "processed_AllFishCombined_unfiltered.csv")
# outputPath = os.path.join(dataPath, "acousticDataFrame.csv")

# Prepare accoustic data for analysis
# acousticData = prepAcousticData(inputPath, outputPath)

# For each fish, extract PCs and scores from them
inputPath = os.path.join(dataPath, "acousticDataFrame.csv")
scorePath = os.path.join(dataPath, "scoreVectors.csv")
pcPath = os.path.join(dataPath, "principalComponents.xlsx")

import pandas as pd
acousticData = pd.read_csv(inputPath)
fishList = ["LT001", "BUR001", "LWF001", "SMB001"]
Z3 = concatFeatures(fishList, acousticData, scorePath, pcPath)

# Inpect the top 3 PCs using 3D scatter plot
plot3DPoints(Z3[["PC1", "PC2", "PC3"]], Z3["species"], "4 Species in the top 3 PC Space")

# Clustering
# clusteringResults = clusterer(Z3)

# Visualize Clustering Results in 3D space
# plotResults(Z3, 
#             clusteringResults["labels"], 
#             clusteringResults["mu"],
#             clusteringResults["sigma"],
#             0, "Clustering on the top 3 PCs")
