## main.py calles other analysis scripts, serving as a wrapper function
import time
progStart = time.time()

## Load Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from datetime import datetime
import os

## Load other scripts
from prepAcousticData import prepAcousticData
from pca import pca
from spectrogram import spectrogram
from hardThreshold import hard_threshold
from HMM import performHMM
from permuteClass import permuteSpecies

## Zoom in the current directory
curWorkingDir = os.getcwd()
print("Current working directory: ", curWorkingDir)
## Reset the working directory to the root
# os.chdir("..\\..\\")
# curWorkingDir = os.getcwd()
# print("Current working directory: ", curWorkingDir)


## Create output directory if it does not exist
today = datetime.today().strftime('%Y-%m-%d')
newResultDir = f"Output\\{today}-Results"
if not os.path.exists(newResultDir):
    os.makedirs(newResultDir, exist_ok=True)
    print(f"New directory created: {newResultDir}")

# Read and clean the processed data
readingStart = time.time()
dataPath = "C:\\Users\\86139\\Desktop\\FishTetherExperiment\\ProcessedData"
inputPath = os.path.join(dataPath, "processed_AllFishCombined_unfiltered.csv")
outputPath = os.path.join(dataPath, "acousticDataFrame.csv")

# Prepare accoustic data for analysis
cleanedData = prepAcousticData(inputPath, outputPath)
X, identifiers, lengthsByFish = cleanedData["X"], cleanedData["identifiers"], cleanedData["lengthsByFish"]
readingEnd = time.time()
readingTime = readingEnd - readingStart

# Optional: Load the already saved data
dataPath = os.path.join(curWorkingDir, "ProcessedData", "acousticDataFrame.csv")
df = pd.read_csv(dataPath)
identifiers = df[["fishNum", "species"]]
X = df.drop(columns=["fishNum", "species"])
lengthsByFish = identifiers.groupby("fishNum").size().tolist()

# Inpect the pooled spectrogram of all fish for PCA reduction
cumLengths = np.cumsum(lengthsByFish)
fishList = identifiers["fishNum"].unique()

specStart = time.time()
spectrogram(X, cumLengths, fishList,\
            saveFig=True, title="Spectrogram of all fish before PCA",
            outputPath=newResultDir)
specEnd = time.time()
specTime = specEnd - specStart

## PCA loading vector matrix V
pcaStart = time.time()
V = pca(X, 0.8, 50, newResultDir)
pcaEnd = time.time()
pcaTime = pcaEnd - pcaStart

## Hard Thresholding Top PCs and comput the Z_hat matrix
V_hard = hard_threshold(V, 0.9)
Z_hat = pd.DataFrame(np.dot(X, V_hard), columns=[f"PC{i+1}" for i in range(V_hard.shape[1])])

## Save V_hard and Z_hat to CSV files
V_hard.to_csv(os.path.join(newResultDir, "V_hard.csv"), index=False)
Z_hat.to_csv(os.path.join(newResultDir, "Z_hat.csv"), index=False)

## Visualize the marginal distribution of the first 3 PCs
Z_hat["species"] = identifiers["species"]
hue="species",
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
plt.savefig(newResultDir + "PC-Projections-of-Raw-Data.png")
plt.show()
Z_hat.drop(columns=["species"], inplace=True)

## Compute X_hat and plot it
specStart = time.time()
X_hat = pd.DataFrame(np.dot(Z_hat, V_hard.T), columns=X.columns)
# spectrogram(X_hat, cumLengths, fishList, \
            # saveFig=True, title="Spectrogram of all fish after PCA",\
            # outputPath=newResultDir)
specEnd = time.time()
specTime = specTime + (specEnd - specStart)

## Plot the Scatter plot of the first 2 PCs
scatterStart = time.time()
plt.Figure(figsize=(8, 6))
sns.scatterplot(x=Z_hat.iloc[:, 0], y=Z_hat.iloc[:, 1], hue=identifiers["species"], alpha=0.3)
plt.title('Scatter plot of PC1 and PC2')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.savefig(os.path.join(newResultDir, "Scatter plot of PC1 and PC2.png"), dpi=300)
plt.show()
scatterEnd = time.time()
scatterTime = scatterEnd - scatterStart

# Perform HMM on the Z_hat data
hmmStart = time.time()
hmmResult = performHMM(Z_hat, lengthsByFish, newResultDir)
hmmPredStates = hmmResult["predStates"]
hmmEnd = time.time()
hmmTime = hmmEnd - hmmStart

# # Permute around state-class assignment, return the assignment with the highest accuracy
assignStart = time.time()
speciesList = ['burbot', 'laketrout', 'lakeWhitefish', 'smallmouthBass']
permuteSpecies(hmmPredStates, speciesList, identifiers, newResultDir)
assignEnd = time.time()
assignTime = assignEnd - assignStart


progEnd = time.time()
totalTime = progEnd - progStart
timeReport = pd.DataFrame({
    "Step": ["Total", "Reading Data", "PCA", "Scatter Plot", "HMM", "Permutation"],
    "Time (s)": [totalTime, readingTime, pcaTime, scatterTime, hmmTime, assignTime]
})
timeReport.to_csv(os.path.join(newResultDir, "TimeReport.csv"), index=False)

print("Program finished in ", round(totalTime, 2), "seconds")
print("Reading data time: ", round(readingTime, 2), "seconds")
print("PCA time: ", round(pcaTime, 2), "seconds")
print("Spectrogram time: ", round(specTime, 2), "seconds")
print("Scatter plot time: ", round(scatterTime, 2), "seconds")
print("HMM time: ", round(hmmTime, 2), "seconds")
print("Permutation time: ", round(assignTime, 2), "seconds")