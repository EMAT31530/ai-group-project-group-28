import math
from random import randint


def cluster_assign(data, centroids):
    # For each point calculate the distance it is from all the centroids
    # and assign the point to the centroid it is closest to
    # We make the arrays for the centroids which starts with each 
    # centroid having no points assigned to it
    assigned_centroids = []
    for point in data:
        dist_from_centroids = [] # Includes indexs
        for index, centroid in enumerate(centroids):
            distance = 0
            for i in range(len(point)):
                distance = distance + math.pow(point[i] - centroid[i], 2)
            distance = math.sqrt(distance)
            dist_from_centroids.append((distance, index))
            sorted_dist_from_centroids = sorted(dist_from_centroids)
            #print(sorted_dist_from_centroids)
            #print(sorted_dist_from_centroids[0][1])
            closest_centroid_index = sorted_dist_from_centroids[0][1]
            # We want our index of the first distance in the list^
        assigned_centroids.append((point, closest_centroid_index))
        # This is now our array of [([point], index), ...] where the index is the index of the centroid
    return assigned_centroids

def avg_of_points(points,k):
    # We compute the average of the points which will be our new centroid(s)
    # We start by creating an array which will hold all of our new centroids
    new_centroids = []
    for j in range(k): # Loops though each centroid
        points_in_centroid_j = []
        for i in range(len(points)): # Find all points in that centroid j
            if points[i][1] == j:
                points_in_centroid_j.append(points[i][0]) # If the point belongs to that centroid then we add it to the array
        print(points_in_centroid_j)
        #new_centroid_j = (sum(points_in_centroid_j)) / len(points_in_centroid_j) # Find the average of those points
        #new_centroids.append(new_centroid_j) # Now we add our new centroid for index j
    #print(new_centroids)  
        

def k_means(data, newpoint, k):
    centroids = []
    # Ranomly pick k centroids to start with
    while len(centroids) < k:
        rand_point = data[randint(0,len(data)-1)]
        if rand_point not in centroids:
            centroids.append(rand_point)
        else:
            continue
    # There is definitely a more efficient way to do this ^
    # We create a while loop which keeps the program running until
    # all points don't change their centroid
    #while 
    # We assign our points to clusters
    assigned_centroids = cluster_assign(data,centroids)
    # We re-estimate our k cluster centroids, by assuming the points have been
    # assigned to the correct centroid. 
    new_centroids = avg_of_points(assigned_centroids,k)
    #print(assigned_centroids)
Data = [[1,8], [2,-14], [4,-1], [8,7], [5,-5], [2,3]]

print(k_means(Data, [5,6], 4))
