import math
from collections import Counter

def knn(data, newpoint, k): 
    # Data shouldn't be indexed 
    # We want to be able to test knn on different values of k so we add it as a parameter
    neighbour_dists = [] # with indices
    # For each point in the data we...
    for index, point in enumerate(data):
        # Calculate the distance between the newpoint and the current
        # point from the data.
        distance = 0
        for i in range(len(point[:-1])):
            distance = distance + math.pow(point[:-1][i] - newpoint[i], 2)
        distance = math.sqrt(distance)
        # Last index is the label so we don't include it in this calculation
        # Takes two points and calculates the distance between them using Euclidean Metric
        
        # Add the distance and the index to neighbour_dists
        neighbour_dists.append((distance, index))
    
    # Sort (in ascending order)
    sorted_neighbour_dists = sorted(neighbour_dists)
    
    # Pick the k closest points
    k_nearest_dists_and_indices = sorted_neighbour_dists[:k]
    
    # Get the labels of the k selected points 
    k_nearest_labels = [data[i][-1] for distance, i in k_nearest_dists_and_indices]

    return Counter(k_nearest_labels).most_common(1)[0][0]
    # Returns the predicted label for our newpoint
    # We want the 1st most common and the zeroth element as we want the most
    # common label, not the amount of times it has occured (which is index 1)

# Test
Data = [[1,1], [2,-1], [4,-1], [8,1], [5,1], [2,1]]
print(knn(Data, [6], 2))


