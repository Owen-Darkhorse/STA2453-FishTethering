## Performs Guassian Mixture Model clustering on the data

def GMM_clustering(df, ):
    import pandas as pd
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import SparsePCA
