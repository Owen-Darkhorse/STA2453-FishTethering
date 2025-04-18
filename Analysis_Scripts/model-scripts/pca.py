# For each fish number, performs sparse PCA to extract features from the fish's data
# Essential features are score vectors in the transformed space
# The extracted features are saved to a new csv file
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def pca(df, recoveryRate, maxComponents=50, outputPath=None):
    '''
    Perform PCA on the given DataFrame and plot the explained variance ratios and loading vectors.
    Args:
        df (pd.DataFrame): The input DataFrame with features.
        recoveryRate (float): The desired recovery rate for variance explained.
        maxComponents (int): The maximum number of components to consider for PCA.
        outputPath (str): The path to save the plots. If None, the plots will not be saved.
    Returns:
        V(pd.DataFrame): The loading vectors of the PCA.
    Side Effects:
        Saves the explained variance ratios and loading vectors plots to the specified output path.
    '''
    # Set all NAs values to 0
    df = df.fillna(0)

    # Standardize the data
    scaler = StandardScaler()
    df = scaler.fit_transform(df)

    # SVD dimensionality reduction
    pca = PCA(n_components=maxComponents, random_state=42)
    pca.fit(df)

    # Get the top PCs that explains the 80% of the variance
    varExplained = np.cumsum(pca.explained_variance_ratio_)
    numPCs = np.argmax(varExplained >= recoveryRate) + 1
    print(f"Number of PCs that explain {recoveryRate *100}% of the variance: {numPCs}")

    
    V = pca.components_[:numPCs].T

    V = pd.DataFrame(V)
    # Z = pd.DataFrame(Z)

    # X_hat = Z @ V.T
    # X_hat = pd.DataFrame(X_hat).reset_index(drop=True)
    # spectrogram(X_hat, saveFig=False, n=X_hat.shape[0])

    # plot the elbow curve
    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize=(12, 6))
    ax1.plot(range(1, len(varExplained) + 1), pca.explained_variance_ratio_)
    ax1.set_title('Elbow curve')
    ax1.set_xlabel('Number of PCs')
    ax1.set_ylabel('Explained variance ratios')
    ax1.axhline(y=pca.explained_variance_ratio_[numPCs-1], color='r', linestyle='--')
    ax1.axvline(x=numPCs, color='g', linestyle='--')

    # Plot the first two loading vectors
    # ax2.figure(figsize=(8, 6))
    ax2.plot(V.iloc[:, 0], label='PC1')
    ax2.plot(V.iloc[:, 1],  label='PC2')
    ax2.plot(V.iloc[:, 2],  label='PC3')
    ax2.plot(V.iloc[:, 3],  label='PC4')
    ax2.axhline(y=0, color='grey', linestyle='--')
    ax2.axhline(y=-0.025, color='grey', linestyle='--')
    ax2.axhline(y=0.025, color='grey', linestyle='--')
    ax2.set_title('Loading vectors') 
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Loading value')
    ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(outputPath, "Principal Components Plots.png"))
    plt.close()

    # Return extracted features
    return V