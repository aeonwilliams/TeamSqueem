import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class FCM:
    '''Class for Fuzzy C-Means. Currently best suited with Pandas DataFrames for data.'''
    def __init__(self,data,c,m,maxiter=100,genCentroids=False,distFunc=False):
        '''data is data you want to cluster, c is number of clusters, m is dimensionality of data'''
        self._data = data       # Data to cluster on
        self._iter = maxiter    # Max iterations
        self._c = c             # Number of clusters
        self._n = len(data)     # Number of items in data set
        self._m = m             # Dimensionality of the data set
        self._A = np.zeros((self._n,self._c))
        
        # Sample centroids from data
        if isinstance(genCentroids,bool):
            if genCentroids == True and type(self._data) == pd.DataFrame:
                self._centroids = data.sample(self._c).copy()
                self._centroids.reset_index(drop=True,inplace=True)
            elif genCentroids == False:
                self._centroids = pd.DataFrame([np.random.uniform(0,1,self._m) for x in range(self._c)])
        elif isinstance(genCentroids,list):
            self._centroids = pd.DataFrame(genCentroids)

        # Function for calculating distance between elements
        if callable(distFunc):
            self._distFunc = distFunc
        else:
            self._distFunc = FCM.__SimpleEuclidean
        
    # Run FCM on the data
    def fit(self):
        '''Runs the algorithm on the data provided during construction. Uses the distance function to calculate distance. Returns the centroids'''
        with np.errstate(divide='ignore',invalid='ignore'):
            for iteration in range(self._iter):
                #Assignment Step
                DTC = np.zeros((self._n,self._c))
                for k in range(self._n):
                    for i in range(self._c):
                        #calculate distance
                        DTC[k,i] = self._distFunc(self._data.iloc[k],self._centroids.iloc[i],self._m)
                #Update A with membership
                for i in range(self._c):
                    for k in range(self._n):
                        sum = 0
                        for j in range(self._c):
                            sum += (DTC[k,i]/DTC[k,j]) ** (2/(self._m-1))
                        self._A[k,i] = 1/sum
                self._A[np.isnan(self._A)] = 0
                #Update
                for i in range(self._c):
                    sumnum = np.zeros(self._m)
                    sumden = 0
                    for k in range(self._n):
                        sumnum += (self._A[k,i] ** self._m) * self._data.iloc[k].values
                        sumden += self._A[k,i] ** self._m
                    self._centroids.iloc[i] = sumnum/sumden
        return self._centroids

    # Plotting the data and centroids
    def plot(self):
        '''Rudimentally plots the data using PyPlot for 2d.'''
        if self._m == 2:
            fig = plt.figure(figsize=(8,8))
            columns = self._data.columns
            plt.scatter(self._data[columns[0]],self._data[columns[1]])
            columns = self._centroids.columns
            plt.scatter(self._centroids[columns[0]],self._centroids[columns[1]])
            plt.show()
            plt.close(fig)
    # Simple distance formula
    def __SimpleEuclidean(a,b,dim):
        '''Calculates distance with Euclidean Distance. Treats the points a and b as an dim-dimensional array.'''
        distance = 0
        for i in range(dim):
            distance += (a[i] - b[i]) ** 2
        return np.sqrt(distance)