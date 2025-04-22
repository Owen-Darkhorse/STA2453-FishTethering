# The script aims to select acoustic data from the preprocessed dataset with class labels

def prep_acoustic_data(inputPath, outputPath):
    import pandas as pd
    import re

    # Load the preprocessed dataset
    df = pd.read_csv(inputPath)

    # Select the columns with acoustic features
    allFreq = [col for col in df.columns if re.match(r'^F\d+', col)]
    requiredColumns = ['fishNum','species'] + allFreq
    df = df[requiredColumns]
    
    # Save the selected data to a new csv file
    df.to_csv(outputPath, index=False)
    return df