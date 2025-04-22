## Performs K-means clustering on the data

def clusterer(df):
    import pandas as pd
    from sklearn.mixture import GaussianMixture
    
    # Perform clustering
    gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=226,
                          max_iter=1000)
    gmm.fit(df)
    labels = gmm.predict(df)

    # Get means and variance matrices
    means = gmm.means_
    covariances = gmm.covariances_

    # Add the labels to the dataframe
    # df['cluster'] = labels

    AIC = gmm.aic(df)
    BIC = gmm.bic(df)
    print("AIC: {0}, BIC: {1}".format(AIC, BIC))

    return {"AIC": AIC,
            "BIC": BIC, 
            "mu": means, 
            "sigma": covariances, 
            "labels": labels}
