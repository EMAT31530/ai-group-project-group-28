import math
from random import randint

def k_means(data, newpoint, k):
    centroids = []
    while len(centroids) < k:
        rand_point = data[randint(0,len(data)-1)]
        if rand_point not in centroids:
            centroids.append(rand_point)
        else:
            continue
    # There is definitely a more efficient way to do this ^
    print(centroids)
    distance = 0
    for point in data:
        for centroid in centroids:
            for i in range(len(point)):
                distance = distance + math.pow(point[i] - newpoint[i], 2)
                distance = math.sqrt(distance)

Data = [[1,8], [2,-14], [4,-1], [8,7], [5,-5], [2,3]]

print(k_means(Data, [5,6], 4))