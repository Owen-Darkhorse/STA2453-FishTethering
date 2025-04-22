# For each fish number, performs sparse PCA to extract features from the fish's data
# Essential features are score vectors in the transformed space
# The extracted features are saved to a new csv file

def extract_feature(fishNum, df):
    import pandas as pd
    from sklearn.decomposition import SparsePCA
    from sklearn.preprocessing import StandardScaler

    # Select the data for the fish
    fishData = df[df['fishNum'] == fishNum]

    # Remove the fish number and species columns
    fishData = fishData.drop(columns=['fishNum', 'species'])

    # Remove the columns and rows with huge proportion of missing values
    fishData = fishData.dropna(axis=1, thresh=0.5*len(fishData))
    fishData = fishData.dropna(axis=0, thresh=0.5*len(fishData.columns))

    # Fill missing values with the mean of the column
    fishData = fishData.fillna(fishData.mean(), axis=1)

    # Standardize the data
    scaler = StandardScaler()
    fishData = scaler.fit_transform(fishData)

    # Perform sparse PCA
    spca = SparsePCA(n_components=10, alpha=0.01)
    spcaResult = spca.fit_transform(fishData)
    print("Explained variance ratio in the top 3 PCs: {0}%".format(round(sum(spcaResult.explained_variance_ratio_[0:3])*100, 2)))
    
    # Get the top 3 PCs and score vectors
    V3 = spcaResult.components_[:3,]
    Z3 = fishData @ V3.T
    Z3.rename(columns={0: "PC1", 1: "PC2", 2: "PC3"}, inplace=True)

    # Return extracted features
    return {"V3": V3, "Z3": Z3}
    