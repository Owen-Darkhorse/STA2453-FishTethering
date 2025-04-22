# The script aims to select acoustic data from the preprocessed dataset with class labels

def prepAcousticData(inputPath, outputPath):
    '''
    Extracts the acoustic features from the preprocessed dataset, impute NA values by 0,
    and saves them to a new csv file.
    Args:
        inputPath (str): Path to the preprocessed dataset.
        outputPath (str): Path to save the selected data.
    Returns:
        dict: A dictionary containing the following keys:
            - "X": DataFrame of acoustic features.
            - "identifiers": DataFrame of fish identifiers (fishNum and species).
            - "lengthsByFish": List of lengths of each fish.
    '''
    import pandas as pd
    import re

    # Load the preprocessed dataset
    df = pd.read_csv(inputPath)

    # Select the columns with acoustic features
    allFreq = [col for col in df.columns if re.match(r'^F\d+', col)]

    X = df[allFreq]
    X.fillna(0, inplace=True)
    identifiers = df[['fishNum', 'species']]
    lengthsByFish = identifiers.groupby("fishNum").size().tolist()

    df = pd.concat([identifiers, X], axis=1)
    
    # Save the selected data to a new csv file
    df.to_csv(outputPath, index=False)
    return {"X": X, "identifiers": identifiers, "lengthsByFish": lengthsByFish}