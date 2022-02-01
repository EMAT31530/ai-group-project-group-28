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
    print(centroids)
    # For each point calculate the distance it is from all the centroids
    # and assign the point to the centroid it is closest to
    for point in data:
        dist_from_centroids = [] # Includes indexs
        for index, centroid in enumerate(centroids):
            distance = 0
            for i in range(len(point)):
                distance = distance + math.pow(point[i] - centroid[i], 2)
            distance = math.sqrt(distance)
            dist_from_centroids.append((distance, index))
            sorted_dist_from_centroids = sorted(dist_from_centroids)
            #sorted_dist_from_centroids[0]
            #nearest_centroid = max(dist_from_centroid)
        print(sorted_dist_from_centroids)
        print('\n')

Data = [[1,8], [2,-14], [4,-1], [8,7], [5,-5], [2,3]]

print(k_means(Data, [5,6], 4))
