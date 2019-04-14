"""
    Winning Python script for EasyMarkit Hackathon by Team Sigma
"""

##Team Sigma - Members: Betty Zhou, Bailey Lei, Alex Pak

# Usage: python sigma_script.py data/train.csv data/test.csv


# import any necessary packages here
#loading libraries
import argparse
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# read in command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("train_file_path") #path of training set
parser.add_argument("test_file_path") #path of test set
args = parser.parse_args()

def onehot_drop(df, column_name):
    for index in column_name:
        one_hot = pd.get_dummies(df[index], prefix = index)
        df = df.drop(index,axis = 1)
        df = df.join(one_hot)
    return df

def fit_train(df):
    train_df = df
    train_clean = onehot_drop(train_df, ['type', 'province'])
    train_clean['cli_area'] = train_clean['cli_area'].map({'Urban':1, 'Rural':0})
    train_clean['pat_area'] = train_clean['pat_area'].map({'Urban':1, 'Rural':0})
    train_clean['gender'] = train_clean['gender'].map({'M':1, 'F':0})

    # convert to datetime
    train_clean['apt_date'] = pd.to_datetime(train_df.apt_date,format='%Y-%m-%d %H:%M:%S', utc =True)
    train_clean['sent_time'] = pd.to_datetime(train_df.sent_time,format='%Y-%m-%d %H:%M', utc =True)
    train_clean['send_time'] = pd.to_datetime(train_df.send_time, format='%H:%M:%S', utc =True).dt.time

    # find time between reminder and appointment
    train_clean['sent_to_apt'] = (train_clean['apt_date'] - train_clean['sent_time']).dt.total_seconds()/3600

    # attributes
    train_clean['apt_month'] = train_clean['apt_date'].dt.month
    train_clean['sent_day_of_week'] = train_clean['sent_time'].dt.day_name()

    # one-hot encoding
    train_clean = onehot_drop(train_clean, ['sent_day_of_week'])

    X = train_clean.iloc[:, 2:]
    y = train_clean.iloc[:,1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)

    X_train_drop = X_train.drop(["apt_type", "apt_date", "sent_time", "send_time", "city", "cli_zip", 'pat_id', 'family_id','clinic'], axis = 1)
    X_test_drop = X_test.drop(["apt_type", "apt_date", "sent_time", "send_time", "city", "cli_zip", 'pat_id', 'family_id','clinic'], axis = 1)

    print("Number of training examples:", len(y_train))
    print("Number of test examples:", len(y_test))

    lg = lgb.LGBMClassifier(silent=False, n_estimators = 2000, max_depth=100)

    lg_model = lg.fit(X_train_drop, y_train)

    print("train accuracy: ", lg.score(X_train_drop, y_train))
    print("test accuracy: ", lg.score(X_test_drop, y_test))

    return lg_model

def predict_test(test_df, lg_model):
    test_clean = onehot_drop(test_df, ['type', 'province'])
    test_clean['cli_area'] = test_clean['cli_area'].map({'Urban':1, 'Rural':0})
    test_clean['pat_area'] = test_clean['pat_area'].map({'Urban':1, 'Rural':0})
    test_clean['gender'] = test_clean['gender'].map({'M':1, 'F':0})

    # convert to datetime
    test_clean['apt_date'] = pd.to_datetime(test_df.apt_date,format='%Y-%m-%d %H:%M:%S', utc =True)
    test_clean['sent_time'] = pd.to_datetime(test_df.sent_time,format='%Y-%m-%d %H:%M', utc =True)
    test_clean['send_time'] = pd.to_datetime(test_df.send_time, format='%H:%M:%S', utc =True).dt.time

    # find time between reminder and appointment
    test_clean['sent_to_apt'] = (test_clean['apt_date'] - test_clean['sent_time']).dt.total_seconds()/3600

    # attributes
    test_clean['apt_month'] = test_clean['apt_date'].dt.month
    test_clean['sent_day_of_week'] = test_clean['sent_time'].dt.day_name()

    # one-hot encoding
    test_clean = onehot_drop(test_clean, ['sent_day_of_week'])
    test_clean_month = onehot_drop(test_clean, ['apt_month'])

    test_final = test_clean.iloc[:, 1:]
    test_final = test_final.drop(["apt_type", "apt_date", "sent_time", "send_time", "city", "cli_zip", 'pat_id', 'family_id','clinic'], axis = 1)

    print("Number of test examples:", len(test_df))
    print("Number of final cleaned test examples:", len(test_final))
    print("test data shape: ", test_final.shape)

    test_clean["response"] = lg_model.predict(test_final)
    df = test_clean[["ReminderId","response"]]
    return df

def write_to_csv(df):
    group_name = "sigma"
    df.to_csv(group_name + "_output.csv", index=False)
    print(group_name + "_output.csv output successful")

def main():
    # loading train and test data
    train_df = pd.read_csv(args.train_file_path)
    test_df = pd.read_csv(args.test_file_path)

    # pre-processing input train and test data for training model
    lg_model = fit_train(train_df)

    #predict and write to new CSV for submission
    df = predict_test(test_df, lg_model)
    write_to_csv(df)

if __name__ == "__main__":
    main()
