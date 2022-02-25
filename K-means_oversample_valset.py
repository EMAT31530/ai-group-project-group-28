
# We oversample our data set.

import math
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from random import randint
import imblearn
from imblearn.over_sampling import RandomOverSampler
import random
  
        
def k_means_oversample_val(centroids,x_val): # centroids must be [centroid2(predict not default), centroid1(predict default)]
    y_pred = [] # our predicted y values for the validation set will be in the same order of the original xvals so we can compare our ypred and ytrue later
    for point in x_val:
        dist_from_centroids = [] # Includes indexs
        for index, centroid in enumerate(centroids):
            distance = 0
            for i in range(len(point)):
                distance = distance + math.pow((point[i] - centroid[i]), 2)
            distance = math.sqrt(distance)
            dist_from_centroids.append((distance, index))
            sorted_dist_from_centroids = sorted(dist_from_centroids)
            closest_centroid_index = sorted_dist_from_centroids[0][1]
            # We want our index of the first distance in the list^
        y_pred.append(closest_centroid_index)
    return(y_pred)




   