from itertools import permutations
import numpy as np
import pandas as pd

def permuteSpecies(stateHat, speciesList, identifiers):
    """
    Permute the estimated species labels at each time point find the species assignment that maximizes accuracy
    """
    allPermutations = np.array(list(permutations([0, 1, 2, 3])))
    accSmry = pd.DataFrame(columns=["State-Class", "accuracy"])

    ## Compute the estimated state sequence on the training dataset
    for perm in allPermutations:
        fishMap = dict(zip(perm.tolist(), speciesList))
        speciesHat = [fishMap[state] for state in stateHat]

        ## Compute the estimated probability for each fish based on the prediction
        identifiers["speciesHat"] = speciesHat

        ## For each fish, compute the proportion of being in each species
        speciesHatCount = identifiers.groupby(["fishNum", "speciesHat"]).size().unstack().fillna(0)
        speciesHatProp = speciesHatCount.div(speciesHatCount.sum(axis=1), axis=0).reset_index("fishNum")
        finalSpeciesHat = speciesHatProp.apply(lambda x: x.loc[["burbot","lakeWhitefish","laketrout","smallmouthBass"]].idxmax(), axis=1).tolist()

        actalSpecies = identifiers.groupby("fishNum")["species"].first().tolist()
        speciesCompare = pd.DataFrame({"actual": actalSpecies, 
                                       "predicted": finalSpeciesHat})
        accuracy = np.mean(speciesCompare["actual"] == speciesCompare["predicted"])

        newRes = pd.DataFrame({"State-Class": [str(fishMap)], "accuracy": [accuracy]})
        accSmry = pd.concat([accSmry, newRes], ignore_index=True)

    ## Find the best permutation
    bestAcc = accSmry["accuracy"].argmax()
    bestPerm = accSmry.iloc[bestAcc, :]
    
    print("State-Species mapping: ", bestPerm[0])
    print("Accuracy: ", round(bestPerm[1]*100, 2), "%\n")

    return bestPerm
BUR = np.full((25, 2), ["BUR001", "burbot"])
LT = np.full((25, 2), ["LT001", "lakeTrout"])
LWF = np.full((25, 2), ["LWF001", "lakeWhitefish"])
SMB = np.full((25, 2), ["SMB001", "burbot"])
identifiers = pd.DataFrame(np.vstack((BUR, LT, LWF, SMB)), columns=["fishNum", "species"])

rng = np.random.default_rng(0)
stateHat = rng.choice([0, 1, 2, 3], size=100)
speciesList = ['burbot', 'laketrout', 'lakeWhitefish', 'smallmouthBass']
permuteSpecies(stateHat, speciesList, identifiers)