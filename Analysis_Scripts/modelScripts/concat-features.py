# Concatenate extracted features from all fish data

def concat_features(fishList, df, outputPath):
    import pandas as pd
    from extract_feature import extract_feature

    # Initialize empty dataframes to store the extracted features
    V3 = pd.DataFrame()
    Z3 = pd.DataFrame()

    # Extract features for each fish
    for fishNum in fishList:
        # Select the data for the fish
        fishFeatures = extract_feature(fishNum, df)
        V3 = V3.append(pd.concat([pd.Series([fishNum]), fishFeatures["V3"]], axis=1))
        Z3 = Z3.append(pd.concat([pd.Series([fishNum]), fishFeatures["Z3"]], axis=1))

    print("Extracted features for {0} fish".format(len(fishList)))

    # Save the extracted features to a new csv file
    V3.to_excel(outputPath, sheet_name="V3", index=False)
    Z3.to_excel(outputPath, sheet_name="Z3", index=False)
    return {"V3": V3, "Z3": Z3}