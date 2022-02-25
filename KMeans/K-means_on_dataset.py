from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from random import randint
import random


df = pd.read_csv('ActualActualData/training_set.csv') # 37 features


def cluster_assign(data, centroids):
    '''For each point calculate the distance it is from all the centroids
       and assign the point to the centroid it is closest to.
       We make the arrays for the centroids which starts with each 
       centroid having no points assigned to it'''
    assigned_points = []
    for point in data: # Loop over each point in the data
        dist_from_centroids = [] # (Includes indexs)
        for index, centroid in enumerate(centroids): # We give each centroid an index and loop over them
            #print('pc', point, centroid)
            distance = 0
            for i in range(len(point)): # Calculate the Euclidean Distance for each of the centroids and that point
                distance = distance + math.pow((point[i] - centroid[i]), 2)
            distance = math.sqrt(distance)
            dist_from_centroids.append((distance, index))
            sorted_dist_from_centroids = sorted(dist_from_centroids)
            closest_centroid_index = sorted_dist_from_centroids[0][1]
            # We want our index of the first distance in the list ^ (we want the closest centroid)
        assigned_points.append((point, closest_centroid_index))
        # This is now our array of [([point], index), ...] where the index is the index of the centroid
    return assigned_points

def avg_of_points(points,k):
    '''We compute the average of the points, which will be our new centroid(s).
       We start by creating an array which will hold all of our new centroids'''
    new_centroids = []
    for j in range(k): # Loops though each centroid
        points_in_centroid_j = []
        for i in range(len(points)): # Find all points in that centroid j
            if points[i][1] == j:
                points_in_centroid_j.append(points[i][0]) # If the point belongs to that centroid then we add it to the array
        #print(points_in_centroid_j)
        #print('j', points_in_centroid_j)
        new_centroid_j = np.mean([points_in_centroid_j], axis = 1)  # Find the average of those points
        new_centroids.append(new_centroid_j) # Now we add our new centroid for index j 
    return(new_centroids)

def same_centroids(new, old):
    '''Compares the new and old centroids. 
       Returns True if they are identical and False otherwise'''
    for i in range(len(new)):
        if new[i] != old[i]:
            return False
        else:
            return True

def k_means(data, newpoint, k, iterations):
    '''A slightly adapted version of K-means which prints out
       how many 1s and 0s each centroid has. We then decide
       which cluster we think should be 1 and which should be 0.'''
    centroids = []
    # Ranomly pick k centroids to start with
    while len(centroids) < k:
        rand_point = data[randint(0,len(data)-1)]
        if rand_point not in centroids:
            centroids.append(rand_point)
        else:
            continue
    # We assign our points to clusters
    assigned_points = cluster_assign(data,centroids)
    old_centroids = centroids
    new_centroids = []
    # We create a while loop which keeps the program running until
    # all points don't change their centroid
    for p in range(iterations):
        new_centroids = avg_of_points(assigned_points,k) # Our new centroids are the average of the points assigned to that cluster
        new_centroids = [new_centroids[i][0] for i in range(len(new_centroids))]
        if np.allclose(new_centroids, old_centroids): # If we have converged then we classify the new point
            dist_from_centroids = [] # Includes indexs
            for index, centroid in enumerate(new_centroids):
                distance = 0
                for i in range(len(newpoint)):
                    distance = distance + math.pow((newpoint[i] - centroid[i]), 2)
                distance = math.sqrt(distance)
                dist_from_centroids.append((distance, index))
                sorted_dist_from_centroids = sorted(dist_from_centroids)
                closest_centroid_index = sorted_dist_from_centroids[0][1]
                assigned_centroid = new_centroids[closest_centroid_index]

            centroid1_labels = []
            centroid2_labels = []
            net_yearly_income = x_train[:,2].tolist() # We use the net yearly income to identify the point, then find its true label
            print(assigned_points)
            for point in assigned_points:
                if point[1] == 0:
                    index = net_yearly_income.index(point[0][2])
                    centroid1_labels.append(x_train[index,-1])
                if point[1] == 1:
                    index = net_yearly_income.index(point[0][2])
                    centroid2_labels.append(x_train[index,-1])
            centroid1_labels = np.array(centroid1_labels).astype(int) # We convert the floats to integers so we can count how many 1s and 0s we have
            centroid2_labels = np.array(centroid2_labels).astype(int)
            print(centroid1_labels)
            print(centroid2_labels)
            print('First centroid has ',np.count_nonzero(centroid1_labels == 1.0) ,'1s and ', np.count_nonzero(centroid1_labels == 0.0),'zeros')
            print('Second centroid has ',np.count_nonzero(centroid2_labels == 1.0) ,'1s and ', np.count_nonzero(centroid2_labels == 0.0),'zeros')
            print('Converged at iteration: ', p)
            return ('Centroids: ', new_centroids ,'Point is assigned to centroid ', assigned_centroid.tolist())
        else:
            assigned_points = cluster_assign(data,new_centroids) # We re-estimate our k cluster centroids, by assuming the 
                                                                 # points have been assigned to the correct centroid.
            old_centroids = new_centroids.copy()

y_train_true = np.array(df.iloc[:,-1]) # y credit card default
x_train = np.array(df.iloc[:,:-1]) # select all rows and all columns from the start up to the last

print(k_means(x_train.tolist(), [33657, 267397], 2, 10))

#print(np.count_nonzero(y_vals == 1)) # 3697 labels of 1
#print(np.count_nonzero(y_vals == 0)) # 41831 labels of 0

# First centroid has  203 1s and  37228 zeros
# Second centroid has  0 1s and  8097 zeros
