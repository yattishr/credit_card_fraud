# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values # take all columns excluding the last column
y = dataset.iloc[:, -1].values # take the last column ONLY

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1)) # set our feature ranges for all columns between 0 (min) & 1 (max)
X = sc.fit_transform(X)

# training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's'] # add markers to the SOM; 0 = circle, s = square
colors = ['r', 'g'] # add color to our markers; r = red, g = green

# loop thru all the rows of the data; i = index of all customerx; x = vectors (each customer observation/row)
for i, x in enumerate(X):
    w = som.winner(x) # get our winning node of customer x
    # plot whether the customer got approval or not, in the middle of the square / node. plot nodes with colors; red or green
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()
    
# finding the frauds
mappings = som.win_map(X)
# frauds = np.concatenate((mappings[(8, 1)], mappings[(6, 8)]), axis = 0)
frauds = mappings[(8, 1)]
frauds = sc.inverse_transform(frauds) # invert our scaling to reveal true values