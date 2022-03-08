# This file contains the training process for how we trained the different training sets using kmeans.
# We want to use this data to determine which cluster we think should be default and which should be no default.

# Check using the actual kmeans module that the algorithm produces similar results
# ADD INFO ABOUT THE Y_TRUE PARAMETER IN LATEX!

# Anything you want to run you will need to uncomment


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from random import randint
import random
from sklearn.metrics import f1_score
from sklearn import metrics
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support

# All the different training sets
df1 = pd.read_csv('ActualActualData/training_set.csv') # No sample
df2 = pd.read_csv('ActualActualData/training_SMOTE.csv') 
df3 = pd.read_csv('ActualActualData/training_RUS.csv') 
df4 = pd.read_csv('ActualActualData/training_ROS.csv') 
df5 = pd.read_csv('ActualActualData/training_cnn_autoencoded_16.csv')
df6 = pd.read_csv('ActualActualData/training_cnn_autoencoded_4.csv')
df7 = pd.read_csv('ActualActualData/training_cnn.csv')

# Validation Set (Autoencoded)
val_auto = pd.read_csv('ActualActualData/validation_autoencoded_16.csv')
val_set = pd.read_csv('ActualActualData/validation_set.csv')

def cluster_assign(data, centroids):
    '''For each point calculate the distance it is from all the centroids
       and assign the point to the centroid it is closest to.
       We make the arrays for the centroids which starts with each 
       centroid having no points assigned to it'''
    assigned_points = []
    for point in data: # Loop over each point in the data
        dist_from_centroids = [] # (Includes indexs)
        for index, centroid in enumerate(centroids): # We give each centroid an index and loop over them
            #print('pc', point, centroid)
            distance = 0
            for i in range(len(point)): # Calculate the Euclidean Distance for each of the centroids and that point
                distance = distance + math.pow((point[i] - centroid[i]), 2)
            distance = math.sqrt(distance)
            dist_from_centroids.append((distance, index))
            sorted_dist_from_centroids = sorted(dist_from_centroids)
            closest_centroid_index = sorted_dist_from_centroids[0][1]
            # We want our index of the first distance in the list ^ (we want the closest centroid)
        assigned_points.append((point, closest_centroid_index))
        # This is now our array of [([point], index), ...] where the index is the index of the centroid
    return assigned_points

def avg_of_points(points,k):
    '''We compute the average of the points, which will be our new centroid(s).
       We start by creating an array which will hold all of our new centroids'''
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
    return(new_centroids)

def same_centroids(new, old):
    '''Compares the new and old centroids. 
       Returns True if they are identical and False otherwise'''
    for i in range(len(new)):
        if new[i] != old[i]:
            return False
        else:
            return True

def k_means(data, newpoint, k, iterations, y_true):
    '''A slightly adapted version of K-means which prints out
       how many 1s and 0s each centroid has. We then decide
       which cluster we think should be 1 and which should be 0.
       The function returns the predicted cluster indexes at the end'''
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
    for p in range(iterations):
        new_centroids = avg_of_points(assigned_points,k) # Our new centroids are the average of the points assigned to that cluster
        #print(new_centroids)
        new_centroids = [new_centroids[i][0] for i in range(len(new_centroids))]
        if np.allclose(new_centroids, old_centroids): # If we have converged then we classify the new point
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
            predicted_cluster_index = [point[1] for point in assigned_points]

            centroid1_labels = [] # True labels of centroid 1 (index 0)
            centroid2_labels = [] # True labels of centroid 2 (index 1)
            centroid3_labels = [] # True labels of centroid 3 (index 2)
            centroid4_labels = [] # True labels of centroid 4 (index 3)
            centroid5_labels = [] # True labels of centroid 5 (index 4)
            centroid6_labels = [] # True labels of centroid 6 (index 5)
            centroid7_labels = [] # True labels of centroid 7 (index 6)
            centroid8_labels = [] # True labels of centroid 8 (index 7)
            centroid9_labels = [] # True labels of centroid 8 (index 7)
            count = 0
            for point in assigned_points:
                if point[1] == 0:
                    centroid1_labels.append(y_true[count]) 
                    count+=1
                if point[1] == 1:
                    centroid2_labels.append(y_true[count])
                    count+=1
                if point[1] == 2:
                    centroid3_labels.append(y_true[count]) 
                    count+=1
                if point[1] == 3:
                    centroid4_labels.append(y_true[count]) 
                    count+=1
                if point[1] == 4:
                    centroid5_labels.append(y_true[count]) 
                    count+=1
                if point[1] == 5:
                    centroid6_labels.append(y_true[count]) 
                    count+=1
                if point[1] == 6:
                    centroid7_labels.append(y_true[count]) 
                    count+=1
                if point[1] == 7:
                    centroid8_labels.append(y_true[count]) 
                    count+=1
            centroid1_labels = np.array(centroid1_labels).astype(int) # We convert the floats to integers so we can count how many 1s and 0s we have
            centroid2_labels = np.array(centroid2_labels).astype(int)
            centroid3_labels = np.array(centroid3_labels).astype(int)
            centroid4_labels = np.array(centroid4_labels).astype(int)
            centroid5_labels = np.array(centroid5_labels).astype(int)
            centroid6_labels = np.array(centroid6_labels).astype(int)
            centroid7_labels = np.array(centroid7_labels).astype(int)
            centroid8_labels = np.array(centroid8_labels).astype(int)
            centroid9_labels = np.array(centroid9_labels).astype(int)
            #print(centroid1_labels)
            #print(centroid2_labels)
            print('First centroid has ',np.count_nonzero(centroid1_labels == 1.0) ,'1s and ', np.count_nonzero(centroid1_labels == 0.0),'zeros')
            print('Second centroid has ',np.count_nonzero(centroid2_labels == 1.0) ,'1s and ', np.count_nonzero(centroid2_labels == 0.0),'zeros')
            print('3rd centroid has ',np.count_nonzero(centroid3_labels == 1.0) ,'1s and ', np.count_nonzero(centroid3_labels == 0.0),'zeros')
            print('4rd centroid has ',np.count_nonzero(centroid4_labels == 1.0) ,'1s and ', np.count_nonzero(centroid4_labels == 0.0),'zeros')
            print('5th centroid has ',np.count_nonzero(centroid5_labels == 1.0) ,'1s and ', np.count_nonzero(centroid5_labels == 0.0),'zeros')
            print('6th centroid has ',np.count_nonzero(centroid6_labels == 1.0) ,'1s and ', np.count_nonzero(centroid6_labels == 0.0),'zeros')
            print('7th centroid has ',np.count_nonzero(centroid7_labels == 1.0) ,'1s and ', np.count_nonzero(centroid7_labels == 0.0),'zeros')
            print('8th centroid has ',np.count_nonzero(centroid8_labels == 1.0) ,'1s and ', np.count_nonzero(centroid8_labels == 0.0),'zeros')
            print('9th centroid has ',np.count_nonzero(centroid9_labels == 1.0) ,'1s and ', np.count_nonzero(centroid9_labels == 0.0),'zeros')
            print('Converged at iteration: ', p)
            print('Centroids: ', new_centroids ,'Point is assigned to centroid ', assigned_centroid.tolist())
            return(predicted_cluster_index[:10]) # If 'None' is returned then it hasn't converged and you need to increase the number of iterations
        else:
            assigned_points = cluster_assign(data,new_centroids) # We re-estimate our k cluster centroids, by assuming the 
                                                                 # points have been assigned to the correct centroid.
            #print(assigned_points[:10])
            old_centroids = new_centroids.copy()



# The original training set has a shape of (21000, 24)

#y1_true = np.array(df1.iloc[:,-1]) # y credit card default
#x1_train = np.array(df1.iloc[:,:-1]) # select all rows and all columns from the start up to the last
#y1_pred = k_means(x1_train.tolist(), [33657, 267397], 2, 10, y1_true)
#print(y1_pred))
# This implies the first centroid is 1 and the second is 0

'''First centroid has  2239 1s and  7376 zeros
Second centroid has  2406 1s and  8979 zeros
Converged at iteration:  3
Centroids:  [array([1.75033959e-01, 3.74726989e-01, 3.23730283e-01, 1.04004160e-04,
       3.26494163e-01, 1.95902236e-01, 1.83338534e-01, 1.79500780e-01,
       1.74498180e-01, 1.69973999e-01, 1.66968279e-01, 1.93703419e-01,
       1.14639504e-01, 1.13458311e-01, 2.02634430e-01, 1.22504758e-01,
       2.92184427e-01, 6.70439034e-03, 3.83667719e-03, 6.08129958e-03,
       8.26653520e-03, 1.15495117e-02, 1.04729568e-02]), array([0.14674456, 0.40570927, 0.24470795, 0.51102328, 0.18291612,
       0.19975406, 0.18907334, 0.18700922, 0.18137022, 0.17703118,
       0.17487923, 0.19059124, 0.1116657 , 0.11120899, 0.19956834,
       0.11921903, 0.28992418, 0.00621967, 0.0031818 , 0.00569393,
       0.00751427, 0.01106253, 0.00978599])] Point is assigned to centroid  [0.1467445646628643, 0.4057092665788318,
       0.24470794905578067, 0.5110232762406676, 0.1829161176987197, 0.19975406236275903, 0.18907334211682975, 0.18700922266140763,
       0.18137022397893457, 0.17703118137902304, 0.17487922705315528, 0.1905912374850702, 0.1116657047916921, 0.11120899246863845,
       0.19956833920257436, 0.11921903412182004, 0.2899241841881974, 0.006219665592343714, 0.0031817956384809676, 0.005693932138083482,
       0.007514273873652075, 0.011062533493069213, 0.009785991270068054]'''

#y1_true = np.array(df1.iloc[:,-1]) # y credit card default
#x1_train = np.array(df1.iloc[:,[0,5]]) # select just indices 0,5
#y1_pred = k_means(x1_train.tolist(), [33657, 267397], 2, 10, y1_true)
#print(y1_pred)
#Suggests centroid 1 is 1s and centroid 2 is 0s

'''First centroid has  3630 1s and  9732 zeros
Second centroid has  1015 1s and  6623 zeros
Converged at iteration:  8
Centroids:  [array([0.08076439, 0.23500973]), array([0.29778275, 0.13322859])]'''

# The SMOTE training set has a shape of (32710, 24)

#y2_true = np.array(df2.iloc[:,-1]) # y credit card default
#x2_train = np.array(df2.iloc[:,:-1]) # select all rows and all columns from the start up to the last
#y2_pred = k_means(x2_train.tolist(), [33657, 267397], 2, 10, y2_true)
#print(y2_pred)
# This suggests first centroid is 1, second is 0

'''First centroid has  7022 1s and  6250 zeros
Second centroid has  9333 1s and  10105 zeros
Converged at iteration:  4
Centroids:  [array([0.14004153, 1.        , 0.28322842, 0.27877087, 0.26903657,
       0.2301811 , 0.217302  , 0.21197883, 0.20383612, 0.1973617 ,
       0.19366655, 0.19285495, 0.11421901, 0.11267359, 0.20169637,
       0.12150938, 0.29156636, 0.00534625, 0.00290752, 0.00513146,
       0.00673001, 0.00950906, 0.008457  ]), array([0.14668545, 0.        , 0.28917094, 0.26637622, 0.23583676,
       0.21612724, 0.19920324, 0.19306873, 0.1872381 , 0.18267377,
       0.17992873, 0.18833287, 0.10952317, 0.11050716, 0.19866701,
       0.11873395, 0.28958428, 0.00536502, 0.00285779, 0.00477461,
       0.00642918, 0.00958548, 0.00834993])] Point is assigned to centroid  [0.14004153462686422, 1.0, 0.2832284173992454, 0.2787708712460896, 0.2690365681235959, 0.23018109734656766, 0.21730199644866652, 0.211978827763337, 0.2038361152489261, 0.1973617047218983, 0.19366654879525655, 0.19285494675821746, 0.11421901302105404, 0.11267359168417108, 0.20169637262252776, 0.12150937589609356, 0.29156635990036134, 0.005346249930783723, 0.0029075161882290155, 0.005131459218150766, 0.006730005862237024, 0.009509063231351396, 0.008457001855400487]'''

#y2_true = np.array(df2.iloc[:,-1]) # y credit card default
#x2_train = np.array(df2.iloc[:,[0,5]]) # select just indices 0,5
#y2_pred = k_means(x2_train.tolist(), [33657, 267397], 2, 50, y2_true)
#print(y2_pred)

#This suggest centroid 1 is 0 and centroid 2 is 1

'''First centroid has  3926 1s and  7628 zeros
Second centroid has  12429 1s and  8727 zeros
Converged at iteration:  9
Centroids:  [array([0.2737626 , 0.13347256]), array([0.07311637, 0.27008427])]'''

# The RUS training set has a shape of (9290, 24)

#y3_true = np.array(df3.iloc[:,-1]) # y credit card default
#x3_train = np.array(df3.iloc[:,:-1]) # select all rows and all columns from the start up to the last
#y3_pred = k_means(x3_train.tolist(), [33657, 267397], 2, 10, y3_true)
#print(y3_pred)
# This suggests centroid 1 is 1 and centroid 2 is 0

'''First centroid has  1972 1s and  1777 zeros
Second centroid has  2673 1s and  2868 zeros
Converged at iteration:  2
Centroids:  [array([0.14334866, 1.        , 0.28425358, 0.2766071 , 0.27202196,
       0.23016804, 0.21840491, 0.21251   , 0.20440117, 0.19679915,
       0.1923713 , 0.19446582, 0.11559755, 0.11339202, 0.20269403,
       0.1219145 , 0.2917826 , 0.00578442, 0.00299893, 0.00558583,
       0.00708535, 0.00998947, 0.00933892]), array([0.14865238, 0.        , 0.29350899, 0.26800217, 0.23539881,
       0.21548457, 0.19976539, 0.19382783, 0.18745714, 0.18278289,
       0.17985923, 0.18921042, 0.11036565, 0.11103887, 0.19907743,
       0.1191505 , 0.28976066, 0.0053784 , 0.00307304, 0.00496863,
       0.00705105, 0.01013237, 0.00901396])] Point is assigned to centroid  [0.14334866402083782, 1.0, 0.2842535787321095, 0.27660709522539345, 0.272021964477886, 0.23016804481194858, 0.218404907975461, 0.21251000266737902, 0.20440117364630733, 0.19679914643905203, 0.19237129901307143, 0.19446582032145593, 0.1155975543047989, 0.11339202477930828, 0.20269403255060306, 0.12191450232708616, 0.2917826049176088, 0.005784416417425297, 0.0029989309999672604, 0.00558583340927276, 0.007085345786251421, 0.009989473666245965, 0.009338919302098264]'''

#y3_true = np.array(df3.iloc[:,-1]) # y credit card default
#x3_train = np.array(df3.iloc[:,[0,5]]) # select just indices 0,5
#y3_pred = k_means(x3_train.tolist(), [33657, 267397], 2, 50, y3_true)
#print(y3_pred)

# This suggests centroid 1 is 0 and centroid 2 is 1
'''First centroid has  2420 1s and  4110 zeros
Second centroid has  2225 1s and  535 zeros
Converged at iteration:  4
Centroids:  [array([0.170325  , 0.15653905]), array([0.09017201, 0.3748913 ])]'''

# The ROS training set has a shape of (32710, 24)

#y4_true = np.array(df4.iloc[:,-1]) # y credit card default
#x4_train = np.array(df4.iloc[:,:-1]) # select all rows and all columns from the start up to the last
#y4_pred = k_means(x4_train.tolist(), [33657, 267397], 2, 10, y4_true)
#print(y4_pred)
# First centroid is 0 and second is 1

'''First centroid has  9380 1s and  10105 zeros
Second centroid has  6975 1s and  6250 zeros
Converged at iteration:  2
Centroids:  [array([0.14885504, 0.        , 0.28675049, 0.2678214 , 0.23623565,
       0.21664357, 0.19993841, 0.1945086 , 0.18795997, 0.18237105,
       0.17879908, 0.18911367, 0.11041467, 0.11096213, 0.19926809,
       0.11913592, 0.28990454, 0.00555825, 0.00289421, 0.00487499,
       0.00663223, 0.00991216, 0.00893185]), array([0.14232121, 1.        , 0.28350347, 0.27810964, 0.27105925,
       0.23062382, 0.21826087, 0.21370132, 0.20454442, 0.19839698,
       0.19314934, 0.19425027, 0.11567037, 0.11335443, 0.20264872,
       0.12234537, 0.29221899, 0.00563517, 0.00302011, 0.00551096,
       0.00715441, 0.00988582, 0.00885979])] Point is assigned to centroid  [0.14232120830228728, 1.0, 0.2835034656584893, 0.27810964083175804, 0.2710592529822028, 0.23062381852552324, 0.2182608695652176, 0.21370132325141727, 0.20454442344045234, 0.19839697542533044, 0.1931493383742927, 0.19425027175363213, 0.1156703726753854, 0.11335442552145011, 0.20264871573881624, 0.12234536638291953, 0.2922189932024609, 0.0056351738549099166, 0.0030201093038496525, 0.005510958994623624, 0.007154410503212977, 0.00988582204741192, 0.008859786951580986]'''

#y4_true = np.array(df4.iloc[:,-1]) # y credit card default
#x4_train = np.array(df4.iloc[:,[0,5]]) # select just indices 0,5
#y4_pred = k_means(x4_train.tolist(), [33657, 267397], 2, 50, y4_true)
#print(y4_pred)
# This suggests the first centroid is 0 and second is 1
'''First centroid has  4235 1s and  7930 zeros
Second centroid has  12120 1s and  8425 zeros
Converged at iteration:  8
Centroids:  [array([0.27172506, 0.13576654]), array([0.07189599, 0.27353127])]'''

# The CNN training set has a shape of ??

#y7_true = np.array(df7.iloc[:,-1]) # y credit card default
#x7_train = np.array(df7.iloc[:,:-1]) # select all rows and all columns from the start up to the last
#y7_pred = k_means(x7_train.tolist(), [33657, 267397], 2, 10, y7_true)
#print(y7_pred)

#First centroid has  2673 1s and  3246 zeros
#Second centroid has  1972 1s and  2093 zeros
# This suggests the first centroid is 0 and the second is 1
'''Converged at iteration:  3
Centroids:  [array([0.142678  , 0.        , 0.29019542, 0.26381145, 0.23630215,
       0.2192938 , 0.20175705, 0.19565805, 0.18884947, 0.18195641,
       0.1794053 , 0.18740835, 0.10868109, 0.1099282 , 0.19744847,
       0.11726631, 0.28840591, 0.00529595, 0.00266258, 0.0044935 ,
       0.00640226, 0.00934025, 0.00870333]), array([0.13425771, 1.        , 0.28487085, 0.27675277, 0.2717182 ,
       0.23377614, 0.22093481, 0.21490775, 0.20578106, 0.19874539,
       0.19308733, 0.19123252, 0.11259613, 0.11181121, 0.20017419,
       0.11991189, 0.29035162, 0.00533843, 0.00275048, 0.00524292,
       0.00658272, 0.00937491, 0.00791001])]'''

#y7_true = np.array(df7.iloc[:,-1]) # y credit card default
#x7_train = np.array(df7.iloc[:,[0,5]]) # select just indices 0,5
#y7_pred = k_means(x7_train.tolist(), [33657, 267397], 2, 50, y7_true)
#print(y7_pred)
# Implies first centroid is 0 and second is 1

'''First centroid has  2398 1s and  4304 zeros
Second centroid has  2247 1s and  1035 zeros
Converged at iteration:  7
Centroids:  [array([0.16186334, 0.15368547]), array([0.09307148, 0.37120658])]'''


# Unfortunately none of the training sets seem to produce 'good' clusters.

# We now try the autoencoded (and oversampled) training set . It has shape (32710, 9)

#y5_true = np.array(df5.iloc[:,-1]) # y credit card default
#x5_train = np.array(df5.iloc[:,:-1]) # select all rows and all columns from the start up to the last
#y5_pred = k_means(x5_train.tolist(), [33657, 267397], 9, 100, y5_true)
#print(y5_pred)

# After trying lots of different values of k, k-means still was not separating the data well. 

# We now try the autoencoded, cnn (16) training set . It has shape (9984, 17)

#y6_true = np.array(df6.iloc[:,-1]) # y credit card default
#x6_train = np.array(df6.iloc[:,:-1]) # select all rows and all columns from the start up to the last
#y6_pred = k_means(x6_train.tolist(), [33657, 267397], 2, 100, y6_true)
#print(y6_pred)

'''First centroid has  2127 1s and  2168 zeros
Second centroid has  2518 1s and  3171 zeros.
Centroids:  [array([0.44789801, 1.27862644, 0.        , 0.54151996, 0.85868582,
       0.        , 1.18330417, 0.83331639, 0.56106454, 0.7401745 ,
       0.45592836, 0.        , 0.41516219, 0.68342996, 0.42797   ,
       0.77646369]), array([0.4992042 , 0.73063958, 0.        , 0.38571217, 0.40618006,
       0.        , 0.93935585, 0.51656305, 0.46160858, 0.29093075,
       0.33756419, 0.        , 0.32282501, 0.48796825, 0.44535196,
       0.31714773])] '''

# This suggests the first centroid represents 1s (defualt) and the second represents 0s (no default)

# Now we calculate the f1 score:
# We use the autoencoded validation set (val_auto)
true_val = np.array(val_auto.iloc[:,-1]) # Val yvals
x_val = np.array(val_auto.iloc[:,:-1]) 


def k_means_val(centroids,x_val): # centroids must be [centroid(predict not default 0), centroid(predict default 1)]
    '''Returns our predicted y_vals'''
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

pred_val = k_means_val([[0.4992042 , 0.73063958, 0.        , 0.38571217, 0.40618006,
       0.        , 0.93935585, 0.51656305, 0.46160858, 0.29093075,
       0.33756419, 0.        , 0.32282501, 0.48796825, 0.44535196,
       0.31714773], [0.44789801, 1.27862644, 0.        , 0.54151996, 0.85868582,
       0.        , 1.18330417, 0.83331639, 0.56106454, 0.7401745 ,
       0.45592836, 0.        , 0.41516219, 0.68342996, 0.42797   ,
       0.77646369]], x_val)

#print(f1_score(true_val, pred_val, average = 'binary')) # We use the df6 centroids here 
# Output: 0.35674157303370785

# Cannot fix error in K-means selected features. So, for now we do it here




#df = pd.read_csv('Amex_Dataset/training_set.csv')
#val = pd.read_csv('Amex_Dataset/validation_set.csv')
#x1_vals = np.array(df.iloc[:,[9,10]])
#y_vals = np.array(df.iloc[:,-1])
#print(k_means(x1_vals.tolist(), [33657, 267397], 2, 50,  y_vals))


#First centroid has  1969 1s and  0 zeros
#Second centroid has  619 1s and  29281 zeros
# Suggests first centroid predicts 1 and second predicts 0
#Converged at iteration:  3
#Centroids = [[0.66046301, 0.        ],[0.16880693, 0.49288979]] # Swapped around

##true_val = np.array(val.iloc[:,-1])
#x_val = np.array(val.iloc[:,[9,10]])
#pred_val = k_means_val(Centroids, x_val)
#print(f1_score(true_val, pred_val, average = 'binary'))




# Back to Taiwan (Indexes: 0 (Limit Balance), 5 (Sep status) on smote

#y2_true = np.array(df2.iloc[:,-1]) # y credit card default
#x2_train = np.array(df2.iloc[:,[0,5]]) # select only index 0 and 5, but all rows
#y2_pred = k_means(x2_train.tolist(), [33657, 267397], 2, 100, y2_true)
#print(y2_pred)

'''First centroid has  12348 1s and  8425 zeros
Second centroid has  4007 1s and  7930 zeros
Converged at iteration:  8
Centroids: [0.07145055, 0.27158833] and [0.27022374, 0.13523837]'''
# The first centroid represents 1, second represents 0
#Centroids = [[0.27022374, 0.13523837], [0.07145055, 0.27158833] ]

#true_val = np.array(val_set.iloc[:,-1]) 
#x_val = np.array(val_set.iloc[:,[0,5]])
#pred_val = k_means_val(Centroids, x_val)

#print(f1_score(true_val, pred_val, average = 'binary')) # 0.421761076836792

# We choose weighted : Calculate metrics for each label, and find their average weighted
#  by support (the number of true instances for each label). This alters ‘macro’ 
# to account for label imbalance; it can result in an F-score that is not between precision and recall.
# Macro is : Calculate metrics for each label, and find their unweighted mean

from sklearn.metrics import precision_score
#prec1=precision_score(true_val, pred_val)
from sklearn.metrics import recall_score
#recall1=recall_score(true_val, pred_val)
#print(prec1)
#print(recall1)
#f11=f1_score(true_val, pred_val)
#print(f11)
#print('weighted)')
#print(precision_recall_fscore_support(true_val, pred_val, average ='weighted')) 

# P,R,Fbeta (0.7449875177247771, 0.5417777777777778, 0.5765415406193515) for k = 2
#print('bin')

#print(precision_recall_fscore_support(true_val, pred_val, average ='binary'))

# Now we try k = 3,4,5,...
# But we saw no improvement so have decided to stick with k=2 as the lower the k, the less computational
# Need to calculate p,r,fbeta for all the different training sets and compare. Should we choose the 2 features for all? 

# For all features:
test_set = pd.read_csv('ActualActualData/test_set.csv') 
test_true_y1 = np.array(test_set.iloc[:,-1])
test_x1 = np.array(test_set.iloc[:,:-1])

# For onlyindex 0,5 features:
test_true_y2 = test_true_y1
test_x2 = np.array(test_set.iloc[:,[0,5]])

# Test for no sample:
# All Features
'''Centroids = [[0.14674456, 0.40570927, 0.24470795, 0.51102328, 0.18291612,
       0.19975406, 0.18907334, 0.18700922, 0.18137022, 0.17703118,
       0.17487923, 0.19059124, 0.1116657 , 0.11120899, 0.19956834,
       0.11921903, 0.28992418, 0.00621967, 0.0031818 , 0.00569393,
       0.00751427, 0.01106253, 0.00978599],[1.75033959e-01, 3.74726989e-01, 3.23730283e-01, 1.04004160e-04,
       3.26494163e-01, 1.95902236e-01, 1.83338534e-01, 1.79500780e-01,
       1.74498180e-01, 1.69973999e-01, 1.66968279e-01, 1.93703419e-01,
       1.14639504e-01, 1.13458311e-01, 2.02634430e-01, 1.22504758e-01,
       2.92184427e-01, 6.70439034e-03, 3.83667719e-03, 6.08129958e-03,
       8.26653520e-03, 1.15495117e-02, 1.04729568e-02]]'''

#pred_test1 = k_means_val(Centroids, test_x1)
#print(precision_recall_fscore_support(test_true_y1, pred_test1, average ='weighted')) 

# 2 features
#Centroids = [[0.29778275, 0.13322859], [0.08076439, 0.23500973]]
#pred_test2 = k_means_val(Centroids, test_x2)
#print(precision_recall_fscore_support(test_true_y2, pred_test2, average ='weighted')) 

# Tests for SMOTE
# All features

'''Centroids = [[0.14668545, 0.        , 0.28917094, 0.26637622, 0.23583676,
       0.21612724, 0.19920324, 0.19306873, 0.1872381 , 0.18267377,
       0.17992873, 0.18833287, 0.10952317, 0.11050716, 0.19866701,
       0.11873395, 0.28958428, 0.00536502, 0.00285779, 0.00477461,
       0.00642918, 0.00958548, 0.00834993], [0.14004153, 1.        , 0.28322842, 0.27877087, 0.26903657,
       0.2301811 , 0.217302  , 0.21197883, 0.20383612, 0.1973617 ,
       0.19366655, 0.19285495, 0.11421901, 0.11267359, 0.20169637,
       0.12150938, 0.29156636, 0.00534625, 0.00290752, 0.00513146,
       0.00673001, 0.00950906, 0.008457  ]]'''

#pred_test1 = k_means_val(Centroids, test_x1)
#print(precision_recall_fscore_support(test_true_y1, pred_test1, average ='weighted'))

#2 features
#Centroids = [[0.2737626 , 0.13347256], [0.07311637, 0.27008427]]
#pred_test2 = k_means_val(Centroids, test_x2)
#print(precision_recall_fscore_support(test_true_y2, pred_test2, average ='weighted')) 

# Tests for Rus
# All features

'''Centroids = [[0.14865238, 0.        , 0.29350899, 0.26800217, 0.23539881,
       0.21548457, 0.19976539, 0.19382783, 0.18745714, 0.18278289,
       0.17985923, 0.18921042, 0.11036565, 0.11103887, 0.19907743,
       0.1191505 , 0.28976066, 0.0053784 , 0.00307304, 0.00496863,
       0.00705105, 0.01013237, 0.00901396], [0.14334866, 1.        , 0.28425358, 0.2766071 , 0.27202196,
       0.23016804, 0.21840491, 0.21251   , 0.20440117, 0.19679915,
       0.1923713 , 0.19446582, 0.11559755, 0.11339202, 0.20269403,
       0.1219145 , 0.2917826 , 0.00578442, 0.00299893, 0.00558583,
       0.00708535, 0.00998947, 0.00933892]]'''

#pred_test1 = k_means_val(Centroids, test_x1)
#print(precision_recall_fscore_support(test_true_y1, pred_test1, average ='weighted'))

# 2 features
#Centroids = [[0.170325  , 0.15653905], [0.09017201, 0.3748913 ]]
#pred_test2 = k_means_val(Centroids, test_x2)
#print(precision_recall_fscore_support(test_true_y2, pred_test2, average ='weighted')) 

# Tests for Ros
# All features

'''Centroids = [[0.14885504, 0.        , 0.28675049, 0.2678214 , 0.23623565,
       0.21664357, 0.19993841, 0.1945086 , 0.18795997, 0.18237105,
       0.17879908, 0.18911367, 0.11041467, 0.11096213, 0.19926809,
       0.11913592, 0.28990454, 0.00555825, 0.00289421, 0.00487499,
       0.00663223, 0.00991216, 0.00893185], [0.14232121, 1.        , 0.28350347, 0.27810964, 0.27105925,
       0.23062382, 0.21826087, 0.21370132, 0.20454442, 0.19839698,
       0.19314934, 0.19425027, 0.11567037, 0.11335443, 0.20264872,
       0.12234537, 0.29221899, 0.00563517, 0.00302011, 0.00551096,
       0.00715441, 0.00988582, 0.00885979]]'''

#pred_test1 = k_means_val(Centroids, test_x1)
#print(precision_recall_fscore_support(test_true_y1, pred_test1, average ='weighted'))

# 2 features

##Centroids = [[0.27172506, 0.13576654], [0.07189599, 0.27353127]]
#red_test2 = k_means_val(Centroids, test_x2)
#print(precision_recall_fscore_support(test_true_y2, pred_test2, average ='weighted'))

# Tests for CNN
# All features

'''Centroids = [[0.142678  , 0.        , 0.29019542, 0.26381145, 0.23630215,
       0.2192938 , 0.20175705, 0.19565805, 0.18884947, 0.18195641,
       0.1794053 , 0.18740835, 0.10868109, 0.1099282 , 0.19744847,
       0.11726631, 0.28840591, 0.00529595, 0.00266258, 0.0044935 ,
       0.00640226, 0.00934025, 0.00870333], [0.13425771, 1.        , 0.28487085, 0.27675277, 0.2717182 ,
       0.23377614, 0.22093481, 0.21490775, 0.20578106, 0.19874539,
       0.19308733, 0.19123252, 0.11259613, 0.11181121, 0.20017419,
       0.11991189, 0.29035162, 0.00533843, 0.00275048, 0.00524292,
       0.00658272, 0.00937491, 0.00791001]]'''

#pred_test1 = k_means_val(Centroids, test_x1)
#print(precision_recall_fscore_support(test_true_y1, pred_test1, average ='weighted'))

# 2 features

#Centroids = [[0.16186334, 0.15368547], [0.09307148, 0.37120658]]
#pred_test2 = k_means_val(Centroids, test_x2)
#print(precision_recall_fscore_support(test_true_y2, pred_test2, average ='weighted'))










            