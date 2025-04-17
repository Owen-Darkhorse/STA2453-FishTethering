# %pip install hmmlearn
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os

def performHMM(Z_hat, lengthsByFish, outputPath=None):
    Z_hat = StandardScaler().fit_transform(Z_hat)
    
    HMM = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=1000, tol=0.01)
    HMM.fit(Z_hat, lengths=lengthsByFish)
    predStates = HMM.predict(Z_hat, lengths=lengthsByFish)
    startProb = HMM.startprob_
    transMat = HMM.transmat_
    means = HMM.means_
    covars = HMM.covars_
    bic = HMM.bic(Z_hat, lengths=lengthsByFish)
    aic = HMM.aic(Z_hat, lengths=lengthsByFish)

    ## Save initial distributions, transition matrix, means and covariances to CSV files
    startProb = pd.DataFrame(startProb, columns=["startProb"])
    transMat = pd.DataFrame(transMat, columns=[f"transMat_{i}" for i in range(transMat.shape[1])])
    # means = np.array([mean[0] for mean in means])
    means = pd.DataFrame(means)
    diagCovars = np.array([np.diag(cov) for cov in covars])
    diagCovars = pd.DataFrame(diagCovars)

    startProb.to_csv(os.path.join(outputPath, "HMMstartProb.csv"), index=False)
    transMat.to_csv(os.path.join(outputPath, "HMMtransMat.csv"), index=False)
    means.to_csv(os.path.join(outputPath, "GMMmeans.csv"), index=False)
    # diagCovars.to_csv(os.path.join(outputPath, "GMMcovars.csv"), index=False)

    print("HMM parameters of HMM: 30 score features, Gaussian Mixture diagonal, before differencing")
    print("Intial distributions: ", startProb, "\n")
    print("Transition matrix: ", transMat, "\n")
    # print("Mean values:", means)
    # print("Covariances:", covars)
    print("Goodness of fit: \n")
    print("BIC:", bic,"\n")
    print("AIC:", aic, "\n")

    ## Save the fitted model to a file
    import pickle
    with open(os.path.join(outputPath, "HMM_model.pkl"), "wb") as f:
        pickle.dump(HMM, f)  

    return {"predStates" : predStates,
            "startProb" : startProb,
            "transMat": transMat,
            "means": means,
            "Covars": covars,
            "bic": bic,
            "aic": aic}