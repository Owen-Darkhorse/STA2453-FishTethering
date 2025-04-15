import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def spectrogram(df, lengthsByFish, fishList, saveFig=True, title=None):
    """
    Function to plot the spectrogram (or heatmap) of the fish's target strength in the first n observations
    Args:
        df: DataFrame with the audio file paths
        fishNum: Fish number to plot the spectrogram
        saveFig: Boolean to save the figure
    Returns:
        None
    """
    # Create the spectrogram
    df.reset_index(drop=True, inplace=True)

    plt.figure(figsize=(12, 8))
    sns.heatmap(df.T, cmap="viridis") 
    plt.title(title)
    plt.vlines(x=lengthsByFish, ymin = -1, ymax = 480, color='red', linestyle='--', label='n')
    plt.xticks(ticks=lengthsByFish, labels=fishList, rotation=45)
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

    if saveFig:
        plt.savefig(f'C:\\Users\\86139\\Desktop\\FishTetherExperiment\\Images\\{title}.png')
    return None

# fishList = ["LT001", "BUR001", "LWF001", "SMB001",
#             "LT002", "BUR002", "LWF002", "SMB002"]
# df = pd.read_csv('C:\\Users\\86139\\Desktop\\FishTetherExperiment\\ProcessedData\\acousticDataFrame.csv')
# for fishNum in fishList:
#     spectrogram(df, fishNum, saveFig=True, n=5000)
