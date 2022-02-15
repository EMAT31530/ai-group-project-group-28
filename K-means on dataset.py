from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from random import randint


df = pd.read_csv('Cleaning_the_data/clean_data.csv')
#print(df.head())
y_true = np.array(df.iloc[:,-1]) # y credit card default
x = np.array(df.iloc[:,:-1]) # select all rows and all columns from the start up to the last
#print(y_true)
x1_test = np.array(df.iloc[:,6]) # 9 and 10, credit_score and previous_defaults
x2_test = np.array(df.iloc[:,2])
#print(x1_test, x2_test)



x1_test = np.delete(x1_test,26662)
x2_test = np.delete(x2_test,26662)
#print(np.where(x2_test == max(x2_test)))
#plt.scatter(x1_test,x2_test)
#plt.show()

x_test = np.array(df.iloc[:,[2,6]])


def cluster_assign(data, centroids):
    # For each point calculate the distance it is from all the centroids
    # and assign the point to the centroid it is closest to
    # We make the arrays for the centroids which starts with each 
    # centroid having no points assigned to it
    assigned_points = []
    for point in data:
        dist_from_centroids = [] # Includes indexs
        for index, centroid in enumerate(centroids):
            #print('pc', point, centroid)
            distance = 0
            for i in range(len(point)):
                distance = distance + math.pow((point[i] - centroid[i]), 2)
            distance = math.sqrt(distance)
            dist_from_centroids.append((distance, index))
            sorted_dist_from_centroids = sorted(dist_from_centroids)
            #print(sorted_dist_from_centroids)
            #print(sorted_dist_from_centroids[0][1])
            closest_centroid_index = sorted_dist_from_centroids[0][1]
            # We want our index of the first distance in the list^
        assigned_points.append((point, closest_centroid_index))
        # This is now our array of [([point], index), ...] where the index is the index of the centroid
    return assigned_points

def avg_of_points(points,k):
    # We compute the average of the points which will be our new centroid(s)
    # We start by creating an array which will hold all of our new centroids
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
        #new_centroid_j = new_centroid_j
        #print(new_centroids) # Why is there no commas? Something to do with np.mean() ?
    #print(list(new_centroids))
    return(new_centroids)

def same_centroids(new, old):
    # I couldn't work our why i couldn't compare the new and old centroids,
    # So I made a function to comoare them instead
    for i in range(len(new)):
        if new[i] != old[i]:
            return False
        else:
            return True
        

def k_means(data, newpoint, k, iterations):
    centroids = []
    # Ranomly pick k centroids to start with
    while len(centroids) < k:
        rand_point = data[randint(0,len(data)-1)]
        #print(np.array(rand_point))
        #print(centroids)
        if rand_point not in centroids:
            centroids.append(rand_point)
        else:
            continue
    #print('original ',centroids)
    # There is definitely a more efficient way to do this ^
    # We assign our points to clusters
    assigned_points = cluster_assign(data,centroids)
    #print(assigned_points)
    old_centroids = centroids
    new_centroids = []
    # We create a while loop which keeps the program running until
    # all points don't change their centroid
    for i in range(iterations):
        new_centroids = avg_of_points(assigned_points,k)
        new_centroids = [new_centroids[i][0] for i in range(len(new_centroids))]
        #print('new', new_centroids)
        #print('old', old_centroids)
        #print(type(new_centroids), type(old_centroids)) #both are lists
        
        if np.allclose(new_centroids, old_centroids):
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
            return ('Centroids: ', new_centroids ,'Point is assigned to centroid ', assigned_centroid.tolist())
        else:
            assigned_points = cluster_assign(data,new_centroids) # We need all the points within the arrays
            old_centroids = new_centroids.copy()

#print(type(x_test))
print(k_means(x_test.tolist(), [33657, 267397], 2, 50))
#print(x_test.tolist())