from hmmlearn import hmm

def performHMM(Z_hat, lengthsByFish):
    HMM = hmm.GaussianHMM(n_components=4, covariance_type="tied", n_iter=1000, tol=0.01)
    HMM.fit(Z_hat, lengths=lengthsByFish)
    predStates = HMM.predict(Z_hat, lengths=lengthsByFish)
    startProb = HMM.startprob_
    transMat = HMM.transmat_
    means = HMM.means_
    covars = HMM.covars_
    bic = HMM.bic(Z_hat, lengths=lengthsByFish)
    aic = HMM.aic(Z_hat, lengths=lengthsByFish)

    print("HMM parameters of HMM: 30 score features, Gaussian Mixture diagonal, before differencing")
    print("Intial distributions: ", startProb)
    print("Transition matrix:", transMat)
    print("Mean values:", means)
    print("Covariances:", covars)
    print("BIC:", bic)
    print("AIC:", aic)

    return {"predStates" : predStates,
            "startProb" : startProb,
            "transMat": transMat,
            "means": means,
            "Covars": covars,
            "bic": bic,
            "aic": aic}

    
