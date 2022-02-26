import math
import numpy as np
from random import randint


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
        if rand_point not in centroids:
            centroids.append(rand_point)
        else:
            continue
    print('original ',centroids)
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
        print('new', new_centroids)
        print('old', old_centroids)
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
        #print(new_centroids)
        #print(new_centroids[0])

        
        # We re-estimate our k cluster centroids, by assuming the points have been
        # assigned to the correct centroid. 
        #print(assigned_points)
    
#Data = [[1,8], [2,-14], [4,-1], [8,7], [5,-5], [2,3]]
Data = [[25,79],[34,51],[22,53],[27,78],[33,59],[33,74],[31,73],[22,57],[35,69],[34,75],[67,51],[54,32],[57,40],[43,47],[50,53],[57,36],[59,35],[52,58],[65,59],[47,50],[49,25],[48,20],[35,14],[33,12],[44,20],[45,5],[38,29],[43,27],[51,8],[46,7]]


print(k_means(Data, [5,6], 4, 50))
