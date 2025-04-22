# Concatenate extracted features from all fish data

def concatFeatures(fishList, df, scorePath, pcPath):
    '''
    Concatenate extracted features from all fish data
    Args:
    fishList: list of strings, fish numbers
    df: pandas dataframe, acoustic data
    csvPath: string, path to save the extracted score vectors
    xlsxPath: string, path to save the extracted loading vectors (PCs)

    Returns:
    Z3: pandas dataframe, extracted score vectors

    Side effects:
    Save the extracted score vectors to a new csv file
    Save the extracted loading vectors to a new xlsx file
    '''
    import pandas as pd
    from extractFeature import extractFeature

    # Initialize empty dataframes to store the extracted features
    # V3 = pd.DataFrame()
    Z3 = pd.DataFrame(columns=["fishNum", "species", "PC1", "PC2", "PC3"])
    V3 = pd.DataFrame(columns=["fishNum", "species", "PC1", "PC2", "PC3"])

    # Extract features for each fish
    for fishNum in fishList:
        # Select the data for the fish
        fishFeatures = extractFeature(fishNum, df)
        species = df[df["fishNum"] == fishNum]["species"].values[0]

        V3New = fishFeatures["V3"]
        V3New["fishNum"] = fishNum
        V3New["species"] = species

        Z3New = fishFeatures["Z3"]
        Z3New["fishNum"] = fishNum
        Z3New["species"] = species

        V3 = pd.concat([V3, V3New], axis=0)
        Z3 = pd.concat([Z3, Z3New], axis=0)        
        
        print("Extracted features for {0} is completed".format(fishNum))

    print("Extracted features for {0} fish".format(len(fishList)))

    # Save the extracted features to a new csv file
    V3.to_csv(pcPath, index=False)
    Z3.to_csv(scorePath, index=False)
    return Z3