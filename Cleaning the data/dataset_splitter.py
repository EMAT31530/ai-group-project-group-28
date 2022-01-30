#Training and validation set creater

import numpy as np
import pandas as pd
import random

df1 = pd.read_csv('clean_data.csv')

def train_validate_test_split(df, train_percent=.7, validate_percent=.15, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test



def csv_creator(train, validate, test):
    train.to_csv('training_set.csv', index=False)
    validate.to_csv('validation_set.csv', index=False)
    test.to_csv('test_set.csv', index=False)


def set_maker(name):
    df1 = pd.read_csv(name)
    
    train, validate, test = train_validate_test_split(df1)

    csv_creator(train, validate, test)
