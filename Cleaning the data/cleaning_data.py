import numpy as np
import pandas as pd

#read the individual csv files
df1 = pd.read_csv('train.csv')

#delete unneccessary columns
df1 = df1.drop(columns=['customer_id','name'])

#fill in the emtpy values in the training set
for col in df1.columns:
        df1[col].fillna(0, inplace=True)

#one-hot-encode the categorical data
final_df1 = pd.get_dummies(df1, columns=['gender', 'owns_car','owns_house','occupation_type'])

final_df1 = final_df1.drop(columns=['owns_car_0'])
q = final_df1.pop('credit_card_default')
final_df1['credit_card_default'] = q


#write the data to new csv files
final_df1.to_csv('clean_data.csv', index=False)



