# preprocess the numerical features into pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import argparse


def preprocess_data(training=True):
    if training:
        data_set = pd.read_csv('data/train.csv')

    else:
        data_set = pd.read_csv('data/test.csv')

    heart = data_set.drop('target', axis=1)

    heart_num = heart.drop(['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'], axis=1)
    heart_cat = heart[['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('std_scaler', StandardScaler()),
    ])

    num_attribs = list(heart_num)
    cat_attribs = list(heart_cat)

    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', OneHotEncoder(categories='auto'), cat_attribs),
    ])

    full_pipeline_fit = full_pipeline.fit(heart)

    return full_pipeline_fit
