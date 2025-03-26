# For each fish number, performs sparse PCA to extract features from the fish's data
# Essential features are score vectors in the transformed space
# The extracted features are saved to a new csv file

def extractFeature(fishNum, df):
    import pandas as pd
    # from sklearn.decomposition import SparsePCA
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Select the data for the fish
    fishData = df[df['fishNum'] == fishNum]

    # Remove the fish number and species columns
    fishData = fishData.drop(columns=['fishNum', 'species'])

    # Remove the columns and rows with huge proportion of missing values
    fishData = fishData.dropna(axis=1, thresh=0.5*len(fishData))
    fishData = fishData.dropna(axis=0, thresh=0.5*len(fishData.columns))

    # Fill missing values with the column mean, column by column 
    colMeans = fishData.mean()
    fishData = fishData.fillna(colMeans)

    # Standardize the data
    scaler = StandardScaler()
    fishData = scaler.fit_transform(fishData)

    # Perform sparse PCA
    # spca = SparsePCA(n_components=3, alpha=0.01)
    # spca.fit(fishData)

    # Perform PCA
    pca = PCA(n_components=3)
    pca.fit(fishData)
    print("Explained variance ratio in the top 3 PCs: {0}%".format(round(sum(pca.explained_variance_ratio_[0:3])*100, 2)))
    
    # Get the top 3 PCs and score vectors
    V3 = pca.components_[:3]
    Z3 = fishData @ V3.T

    V3 = pd.DataFrame(V3.T, columns=["PC1", "PC2", "PC3"])
    Z3 = pd.DataFrame(Z3, columns=["PC1", "PC2", "PC3"])

    # Return extracted features
    return {"V3": V3, "Z3": Z3}
    