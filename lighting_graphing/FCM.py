import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class FCM:
    '''Class for Fuzzy C-Means. Currently best suited with Pandas DataFrames for data.'''
    def __init__(self,data,c,dim,m=2,maxiter=100,genCentroids=False,distFunc=False):
        '''data is data you want to cluster, c is number of clusters, m is dimensionality of data'''
        self._data = data       # Data to cluster on
        self._iter = maxiter    # Max iterations
        self._c = c             # Number of clusters
        self._n = len(data)     # Number of items in data set
        self._dim = dim         # Dimensionality of the data set
        self._m = m             # Fuzzifier
        self._A = np.zeros((self._n,self._c))

        # Sample centroids from data
        if isinstance(genCentroids,bool):
            if self._c <= len(self._data):
                if genCentroids == True and type(self._data) == pd.DataFrame:
                    self._centroids = data.sample(self._c).copy()
                    self._centroids.reset_index(drop=True,inplace=True)
                elif genCentroids == False:
                    self._centroids = pd.DataFrame([np.random.uniform(0,1,self._dim) for x in range(self._c)],columns=['Long','Lat'])
            else:
                self._centroids = pd.DataFrame([np.random.uniform(0,1,self._dim) for x in range(self._c)],columns=['Long','Lat'])
        elif isinstance(genCentroids,list):
            self._centroids = pd.DataFrame(genCentroids)
        elif isinstance(genCentroids,pd.DataFrame):
            self._centroids = genCentroids

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
                        DTC[k,i] = self._distFunc(self._data.iloc[k],self._centroids.iloc[i],self._dim) #dimensionality m
                #Update A with membership
                for i in range(self._c):
                    for k in range(self._n):
                        sum = 0
                        for j in range(self._c):
                            if self._m == 2:
                                sum += (DTC[k,i]/DTC[k,j]) ** 2 #parameter m
                            else:
                                sum += (DTC[k,i]/DTC[k,j]) ** (2/(self._m-1)) #parameter m
                        self._A[k,i] = 1/sum
                self._A[np.isnan(self._A)] = 0
                #Update
                for i in range(self._c):
                    sumnum = np.zeros(self._dim) #dimensionality m
                    sumden = 0
                    for k in range(self._n):
                        sumnum += (self._A[k,i] ** self._m) * self._data.iloc[k].values #parameter m
                        sumden += self._A[k,i] ** self._m #parameter m
                    self._centroids.iloc[i] = sumnum/sumden
        return self._centroids

    # Plotting the data and centroids
    def plot(self):
        '''Rudimentally plots the data using PyPlot for 2d.'''
        if self._dim == 2:
            fig = plt.figure(figsize=(8,8))
            columns = self._data.columns
            plt.scatter(self._data[columns[0]],self._data[columns[1]])
            columns = self._centroids.columns
            plt.scatter(self._centroids[columns[0]],self._centroids[columns[1]])
            plt.show()
            plt.close(fig)

    # De-Fuzzify the data and assign each point to a centroid
    def classify(self,how='max',threshold='.6'):
        '''Classify each point in the data set with it's membership to the centroids'''
        ownership = [0 for x in range(self._n)]
        for x in range(self._n):
            ownership[x] = list(self._A[x]).index(max(self._A[x]))
        return ownership

    # Simple distance formula
    def __SimpleEuclidean(a,b,dim):
        '''Calculates distance with Euclidean Distance. Treats the points a and b as an dim-dimensional array.'''
        distance = 0
        for i in range(dim):
            distance += (a[i] - b[i]) ** 2
        return np.sqrt(distance)

    def __l2Distance(a,b):
        '''Calculates l2 distance between a and b'''
        return np.linalg.norm(a-b)
