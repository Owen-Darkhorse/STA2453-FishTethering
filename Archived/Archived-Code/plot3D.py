## Create a 3D visualization of data points colored by their species labels

import numpy as np
import numpy.linalg as linalg
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot3DPoints(X, y, title):
    '''
    Create a 3D scatter plot for points in X colored by their species labels in y
    Args:
    X: 2D numpy array of shape (n_samples, 3)
    y: 1D numpy array of shape (n_samples,)
    title: string, title of the plot
    '''
    # A list of availabel colors and markers to use
    colors = plt.get_cmap('tab10').colors
    markers = ['o', '^', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    
    # Mapping species labels to colors and markers
    unique_labels = pd.Series(y).unique()
    color_map = dict(zip(unique_labels, colors[:len(unique_labels)]))
    marker_map = dict(zip(unique_labels, markers[:len(unique_labels)]))

    colnames = X.columns

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    for i, label in enumerate(unique_labels):
        ax.scatter(X.loc[y == label, colnames[0]], X.loc[y == label,  colnames[1]], X.loc[y == label,  colnames[2]], c=color_map[label], marker=marker_map[label], label=str(label))
    
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.set_title(title)
    ax.legend()
    plt.show()

# Use case: random number
# X = np.random.rand(150, 3)
# y = pd.Series(["LWF"]*50 + ["LT"]*50 + ["SMB"]*50)
# plot3DPoints(X, y, '3D Scatter Plot')

# Cached File
# Z3 = pd.read_csv("C:\\Users\\86139\\Desktop\\FishTetherExperiment\\ProcessedData\\scoreVectors.csv")
# plot3DPoints(Z3[["PC1", "PC2", "PC3"]], Z3["fishNum"], "Top 3 PCs")

#marker=[marker_map[label] for label in y], 
               #label=[str(label) for label in unique_labels]
# your ellispsoid and center in matrix form
# A = np.array([[1,0,0],[0,2,0],[0,0,2]])
# center = [0,0,0]

# # find the rotation matrix and radii of the axes
# U, s, rotation = linalg.svd(A)
# radii = 1.0/np.sqrt(s)

# # now carry on with EOL's answer
# u = np.linspace(0.0, 2.0 * np.pi, 100)
# v = np.linspace(0.0, np.pi, 100)
# x = radii[0] * np.outer(np.cos(u), np.sin(v))
# y = radii[1] * np.outer(np.sin(u), np.sin(v))
# z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
# for i in range(len(x)):
#     for j in range(len(x)):
#         [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center

# # plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)
# plt.show()