import collections
import numpy as np
import pandas as pd

#read the individual csv files
df1 = pd.read_csv('train.csv')

#delete unneccessary columns
df1 = df1.drop(columns=['customer_id','name'])

##Filling in empty values ##########


#OWNS CAR
df1['owns_car'].fillna(0, inplace=True)

#CHILDREN
children_mean = int(df1['no_of_children'].mean())
df1['no_of_children'].fillna(children_mean, inplace=True)

#Migrant
df1['migrant_worker'].fillna(0, inplace=True)

#debt
debt_median = df1['yearly_debt_payments'].median()
df1['yearly_debt_payments'].fillna(debt_median, inplace=True)

#employment
days_worked = df1['no_of_days_employed'].median()
df1['no_of_days_employed'].fillna(days_worked, inplace=True)

#total family
family_mean = int(df1['total_family_members'].mean())
df1['total_family_members'].fillna(family_mean, inplace=True)


#credit score
credit_score = int(df1['credit_score'].median())
df1['credit_score'].fillna(credit_score, inplace=True)






#one-hot-encode the categorical data

final_df1 = pd.get_dummies(df1, columns=['gender', 'owns_car','owns_house','occupation_type'])
final_df1 = final_df1.drop(columns=['owns_car_0'])

q = final_df1.pop('credit_card_default')
final_df1['credit_card_default'] = q


#write the data to new csv files
final_df1.to_csv('clean_data.csv', index=False)



