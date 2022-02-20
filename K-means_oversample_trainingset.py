# We oversample our data set, train it on kmeans and use the centroids to test it on our validation set

import math
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from random import randint
import imblearn
from imblearn.over_sampling import RandomOverSampler
import random
import matplotlib.pyplot as plt
import seaborn as sns

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
        

def k_means_oversample(allData, data, y_vals, newpoint, k, iterations, samplingStrategy):
    '''Oversamples our data (from 45528 samples to 62746 samples if we use a
    sampling strategy of 0.5) and then applies kmeans. '''
    random.seed(0)
    oversample = RandomOverSampler(sampling_strategy = samplingStrategy) # e.g sampling strategy 0.5 would mean that if the majority
                                                                         # class had 1,000 examples and the minority class had 100, 
                                                                         # the transformed dataset would have 500 examples of the minority class.
    data, y_vals = oversample.fit_resample(data, y_vals) # our new data and new y vals
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
    for i in range(iterations):
        new_centroids = avg_of_points(assigned_points,k)
        new_centroids = [new_centroids[i][0] for i in range(len(new_centroids))]
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
            centroid1_labels = []
            centroid2_labels = []
            net_yearly_income = [point[2] for point in allData]
            for point in assigned_points:
                if point[1] == 0:
                    index = net_yearly_income.index(point[0][0]) # change this each time depending where the net yearly income feature is in the xvals
                    centroid1_labels.append(y_vals[index])
                if point[1] == 1:

                    index = net_yearly_income.index(point[0][0])
                    centroid2_labels.append(y_vals[index])
            centroid1_labels = np.array(centroid1_labels).astype(int)
            centroid2_labels = np.array(centroid2_labels).astype(int)
            print('First centroid has ',np.count_nonzero(centroid1_labels == 1.0) ,'1s and ', np.count_nonzero(centroid1_labels == 0.0),'zeros')
            print('Second centroid has ',np.count_nonzero(centroid2_labels == 1.0) ,'1s and ', np.count_nonzero(centroid2_labels == 0.0),'zeros')
            #print(centroid1_labels)
            #print(centroid2_labels)
            return ('Centroids: ', new_centroids ,'Point is assigned to centroid ', assigned_centroid.tolist())
        else:
            assigned_points = cluster_assign(data,new_centroids) # We need all the points within the arrays
            old_centroids = new_centroids.copy()

df = pd.read_csv('CleaningTheData/ActualData/training_set.csv')
val = pd.read_csv('CleaningTheData/ActualData/validation_set.csv')
allData = np.array(df.iloc[:,:36])

# For all features: x_vals = np.array(df.iloc[:,:36])
# First centroid has  13036 1s and  9805 zeros
# Second centroid has  16237 1s and  19484 zeros
#x_vals = np.array(df.iloc[:,:36])
#y_vals = np.array(df.iloc[:,-1])

# For features 9,10 (credit score and previous defaults)
# For now we also have to incude net yearly income to count the 0s and 1s for each cluster- I should find a better way of doing this
x_vals = np.array(df.iloc[:,[2,9,10]])
y_vals = np.array(df.iloc[:,-1])
# First centroid has  22258 1s and  11 zeros [0.00157757, 0.16860725, 0.49369078]
# Second centroid has  7014 1s and  29279 zeros [0.00121867, 0.61232727, 0.        ]
# This suggests the first centoid contains points that mostly were defaulted (0) and the second centroid mostly contains points that were not defaulted

#print(k_means_oversample(allData.tolist(), x_vals.tolist(), y_vals, [33657, 267397], 2, 50, 'minority'))

# Now we calculate the F1 score
# We need an array for the true yvals and an array for the predicted yvals to calculater this
# So, we create a function which takes our centroids that we got from k_means and predicts which centroid each point in our validation set would be assigned to

def k_means_oversample_val(centroids,x_val): # centroids must be [centroid2(predict not default), centroid1(predict default)]
    ' Returns our predicted y_vals'
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

centroids = [[0.00121867, 0.61232727, 0.        ], [0.00157757, 0.16860725, 0.49369078]]
y_true = np.array(val.iloc[:,-1]) # Validation set
x_val = np.array(val.iloc[:,[2,9,10]]) # xvals from the validation set

y_pred = k_means_oversample_val(centroids, x_val)

#print(f1_score(y_true, y_pred, average = 'binary')) # 0.8650306748466258

cf_matrix = confusion_matrix(y_true, y_pred)
# [[6275    0]
# [ 132  423]]
# We can plot this using the code below

ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

ax.set_title('Credit Default Confusion Matrix \n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()

# Now we add the prev default in 6 months feature

