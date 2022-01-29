from collections import Counter
import math

def mode(labels):
    return Counter(labels).most_common(1)[0][0]

def euclidean_distance(point1, point2):
    sum_squared_distance = 0
    for i in range(len(point1)):
        sum_squared_distance += math.pow(point1[i] - point2[i], 2)
    return math.sqrt(sum_squared_distance)

def knn(data, newpoint, k, distance_fn):
    neighbor_distances_and_indices = []
    
    # For each example in the data
    for index, example in enumerate(data):
        # Calculate the distance between the newpoint example and the current
        # example from the data.
        distance = euclidean_distance(example[:-1], newpoint) 
        # example slice is all up until one before the end
        # takes two points and returns the distance between them using Euclidean Metric
        
        # Add the distance and the index of the example to an ordered collection
        neighbor_distances_and_indices.append((distance, index))
    
    # Sort the ordered collection of distances and indices from
    # smallest to largest (in ascending order) by the distances
    sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices)
    
    # Pick the first K entries from the sorted collection
    k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:k]
    
    # Get the labels of the selected K entries
    k_nearest_labels = [data[i][-1] for distance, i in k_nearest_distances_and_indices]

    return k_nearest_distances_and_indices , mode(k_nearest_labels)





