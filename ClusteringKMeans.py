# In this case study we are generating the dataset at run time randomly and apply user defined K-Mean algorithm.

import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt


def MarvellousKMean():
    Border = "-"*40
    print(Border)
    # Set three centers, the model should predict the similar results
    center_1 = np.array([1,1])
    print(center_1)
    print(Border)
    
    center_2 = np.array([5,5])
    print(center_2)
    print(Border)

    center_3 = np.array([8,1])
    print(center_3)
    print(Border)

    # Generate random data and center it to the three centers

    data_1 = np.random.randn(7,2) + center_1
    print("Element of first cluster with size"+str(len(data_1)))
    print(data_1)
    print(Border)

    data_2 = np.random.randn(7,2) + center_2
    print("Elements of second cluster with size"+str(len(data_2)))
    print(data_2)
    print(Border)

    data_3 = np.random.randn(7,2) + center_3
    print("Elements of third cluster with size" +str(len(data_3)))
    print(data_3)
    print(Border)

    data = np.concatenate((data_1,data_2,data_3),axis = 0)
    print("Size of Complete dataset" +str(len(data)))
    print(data)
    print(Border)

    plt.scatter(data[:,0],data[:,1], s=7)
    plt.title("Marvellous infosystem : Input Dataset")
    plt.show()
    print(Border)

    # Number of clusters
    k = 3

    # Number of training data

    n = data.shape[0]
    print("Total number of elements are",n)
    print(Border)

    # Number of features in the data

    c = data.shape[1]
    print("Total number of features are",c)
    print(Border)

    # Generate random centers, here we use sigma and mean represent the whole data
    mean = np.mean(data,axis = 0)
    print("value of mean",mean)
    print(Border)

    # calculate standard deviation
    std = np.std(data,axis =0)
    print("value of std",std)
    print(Border)

    centers = np.random.randn(k,c)*std + mean
    print("Random points are",centers)
    print(Border)

    # plot the data and the centers generated as random
    plt.scatter(data[:,0],data[:,1], c='r',s=7)
    plt.scatter(centers[:,0],centers[:,1],marker = '*',c='g',s=150)
    plt.title("Marvellous Infosystem : Input Dataset with random centroid *")
    plt.show()
    print(Border)

    centers_old = np.zeros(centers.shape) # to store old centers
    centers_new = deepcopy(centers) # store new centers

    print("Values of old centroids")
    print(centers_old)
    print(Border)

    print("Values of new centroids")
    print(centers_new)
    print(Border)

    data.shape
    cluster = np.zeros(n)
    distances = np.zeros((n,k))

    print("Initial distances are")
    print(distances)
    print(Border)

    error = np.linalg.norm(centers_new - centers_old)
    print("value of error is",error)
    # when, after an update, the estimate of that centerstays the same, exit loop

    while error != 0:
        print("value of error is ",error)
        # Measure the distance to every center
        print("Measure the distanc to every center")
        for i in range (k):
            print("Iterationnumber",i)
            distances[:,i] = np.linalg.norm(data - centers[i], axis= 1)

        # Assign all training data to closest center
        clusters = np.argmin(distances, axis = 1)

        centers_old = deepcopy(centers_new)

        # Calculate mean for every cluster and update the center
        for i in range(k):
            centers_new[i]  = np.mean(data[clusters == i], axis =0)
        error = np.linalg.norm(centers_new - centers_old)
    # end of while 
     
    #centers_new

    # plot the data and the centers generated as random

    plt.scatter(data[:,0],data[:,1],s=7)
    plt.scatter(centers_new[:,0],centers_new[:,1],marker ='*', c='g',s=150)
    plt.title("Marvellous Infosystem : Final data with centroid")
    plt.show()

def main():
        print("----Kaustubh Wani----")
        print("Unsupervised Machine Learning")
        print("Clustering using K Mean Algorithm")
        MarvellousKMean()

if __name__ == "__main__":
    main()