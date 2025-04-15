## main.py calles other analysis scripts, serving as a wrapper function
## Load Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns


import os

## Load other scripts
from prepAcousticData import prepAcousticData
from pca import pca
from spectrogram import spectrogram
from hardThreshold import hard_threshold
from HMM import performHMM
from permuteClass import permuteSpecies

# from concatFeatures import concatFeatures
# from plot3D import plot3DPoints
# from clustering import clusterer
# from visualizeClusters import plotResults


# dataPath = "C:\\Users\\86139\\Desktop\\FishTetherExperiment\\ProcessedData"
# inputPath = os.path.join(dataPath, "processed_AllFishCombined_unfiltered.csv")
# outputPath = os.path.join(dataPath, "acousticDataFrame.csv")

# # Prepare accoustic data for analysis
# cleanedData = prepAcousticData(inputPath, outputPath)
# X, identfiers, lengthsByFish = cleanedData["X"], cleanedData["identifiers"], cleanedData["lengthsByFish"]

# # Load the data
dataPath = "C:\\Users\\86139\\Desktop\\FishTetherExperiment\\ProcessedData\\acousticDataFrame.csv"
df = pd.read_csv(dataPath, nrows=2000)
identifiers = df[["fishNum", "species"]]
X = df.drop(columns=["fishNum", "species"])
lengthsByFish = identifiers.groupby("fishNum").size().tolist()
# Inpect the pooled spectrogram of all fish for PCA reduction
cumLengths = np.cumsum(lengthsByFish)
fishList = identifiers["fishNum"].unique()
# spectrogram(X, cumLengths, fishList, saveFig=True, title="Spectrogram of all fish before PCA")

## PCA loading vector matrix V
V = pca(X, 0.8, 50)

## Hard Thresholding Top PCs and comput the Z_hat matrix
V_hard = hard_threshold(V, 0.9)
Z_hat = pd.DataFrame(np.dot(X, V_hard), columns=[f"PC{i+1}" for i in range(V_hard.shape[1])])

## Visualize the marginal distribution of the first 3 PCs
Z_hat["species"] = identifiers["species"]
#hue="species",
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
sns.histplot(Z_hat.iloc[:, 0], bins=100, alpha=0.4)
plt.title('PC1')
plt.subplot(1, 3, 2)
sns.histplot(Z_hat.iloc[:, 1], bins=100, alpha=0.4)
plt.title('PC2')
plt.subplot(1, 3, 3)
sns.histplot(Z_hat.iloc[:, 2], bins=100, alpha=0.4)
plt.title('PC3')
plt.tight_layout()
plt.show()

## Compute X_hat and plot it
Z_hat.drop(columns=["species"], inplace=True)
X_hat = np.dot(Z_hat, V_hard.T)
# spectrogram(X_hat, cumLengths, fishList, saveFig=True, title="Spectrogram of all fish after PCA")

# Perform HMM on the Z_hat data

# # Permute around state-class assignment, return the assignment with the highest accuracy
speciesList = ['burbot', 'laketrout', 'lakeWhitefish', 'smallmouthBass']
permuteSpecies(HMM, Z_hat, lengthsByFish, identifiers)

# Clustering
# clusteringResults = clusterer(Z3)

# Visualize Clustering Results in 3D space
# plotResults(Z3, 
#             clusteringResults["labels"], 
#             clusteringResults["mu"],
#             clusteringResults["sigma"],
#             0, "Clustering on the top 3 PCs")
