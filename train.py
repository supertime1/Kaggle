import argparse
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from preprocess_data import preprocess_data
from model import models
from sklearn.metrics import f1_score, make_scorer


def train(model_name):
    """

    :param model_name: model type (e.g. random forest) for training
    :return: trained model with randomized search and cross validation
    """
    train_set = pd.read_csv('data/train.csv')
    print('Loading data file successfully!\n')
    heart_labels = train_set['target'].copy()
    heart = train_set.drop('target', axis=1)

    full_pipeline_fit = preprocess_data(training=True)
    heart_prepared = full_pipeline_fit.transform(heart)

    model_dic = {'rf': [models.random_forest()[0], models.random_forest()[1]],
                 'svc': [models.svc()[0], models.svc()[1]]
                 }

    model, config_file = model_dic[model_name]

    print('Current training params are:')
    print(config_file)
    print('\n')

    f1 = make_scorer(f1_score, average='macro')

    grid_search = RandomizedSearchCV(model, config_file, n_iter=1, cv=10, scoring=f1)
    grid_search.fit(heart_prepared, heart_labels)

    print('Best parameters of the training model:')
    print(grid_search.best_params_)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rf', action='store_true', help='select Random Forest as training model')
    parser.add_argument('--svc', action='store_true', help='select Support Vector Machine as training model')
    parser.add_argument('--tree', action='store_true', help='select Decision Tree as training model')
    parser.add_argument('--xgb', action='store_true', help='select XGBoost as training model')
    parser.add_argument('--lin', action='store_true', help='select Linear Regression as training model')

    args = parser.parse_args()
    # set a default model type

    model_name = ('rf' if args.rf else
                  'svc' if args.svc else
                  'tree' if args.tree else
                  'xgb' if args.xgb else
                  'lin' if args.lin else
                  'rf')
    print('Start training...\n')
    train(model_name)
