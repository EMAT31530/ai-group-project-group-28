import math
from random import randint

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
    print(assigned_centroids)
Data = [[1,8], [2,-14], [4,-1], [8,7], [5,-5], [2,3]]

print(k_means(Data, [5,6], 4))
