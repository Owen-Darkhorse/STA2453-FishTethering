import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def spectrogram(df, fishNum, saveFig=True, n=1000):
    """
    Function to plot the spectrogram (or heatmap) of the fish's target strength in the first n observations
    Args:
        df: DataFrame with the audio file paths
        fishNum: Fish number to plot the spectrogram
        saveFig: Boolean to save the figure
    Returns:
        None
    """
    # Get the audio file
    TS = df.loc[df['fishNum'] == fishNum,:]
    TS.drop(columns=['fishNum', 'species'], inplace=True)
    TS = TS.loc[0:min(n,TS.shape[1]),:]

    # Create the spectrogram
    plt.figure(figsize=(12, 8))
    sns.heatmap(TS.T, cmap="viridis") 
    plt.title('Spectrogram of Fish ' + str(fishNum))
    plt.tight_layout()
    if saveFig:
        plt.savefig('C:\\Users\\86139\\Desktop\\FishTetherExperiment\\Images\\spectrogram_fish_' + str(fishNum) + '.png')
    plt.show()

    return None

fishList = ["LT001", "BUR001", "LWF001", "SMB001",
            "LT002", "BUR002", "LWF002", "SMB002"]
df = pd.read_csv('C:\\Users\\86139\\Desktop\\FishTetherExperiment\\ProcessedData\\acousticDataFrame.csv')
for fishNum in fishList:
    spectrogram(df, fishNum, saveFig=True, n=5000)
