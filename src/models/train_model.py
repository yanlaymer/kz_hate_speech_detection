import click
import pandas as pd
import lightgbm as lgb
import joblib as jb

from typing import List
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import json


@click.command()
@click.argument("input_paths", type=click.Path(exists=True), nargs=2)
@click.argument("output_path", type=click.Path(), nargs=2)
def train(input_paths: List[str], output_path: List[str]):
    train_df = pd.read_csv(input_paths[0])
    test_df = pd.read_csv(input_paths[1])

    X_train = train_df.drop('metka', axis=1)
    y_train = train_df['metka']
    X_holdout = test_df.drop('metka', axis=1)
    y_holdout = test_df['metka']

    w2v = TfidfVectorizer(max_features=600)
    X_train = w2v.fit_transform(X_train)
    X_holdout = w2v.transform(X_holdout)

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_holdout, y_holdout, reference=lgb_train)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'l1'},
        'max_depth': 11,
        'num_leaves': 150,
        'learning_rate': 0.25,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'n_estimators': 1000,
        'bagging_freq': 2,
        'verbose': -1,
        'class_weight': {0: 1.0, 1: 4.0}
    }
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=200,
                    valid_sets=lgb_eval,
                    verbose_eval=False,
                    early_stopping_rounds=30)  # categorical_feature=['building_type']
    jb.dump(gbm, output_path[0])

    y_predicted = gbm.predict(X_holdout, num_iteration=gbm.best_iteration)
    score = dict(
        accuracy = accuracy_score(y_holdout, y_predicted),
        precision = precision_score(y_holdout, y_predicted),
        recall = recall_score(y_holdout, y_predicted),
        f1 = f1_score(y_holdout, y_predicted)
    )

    with open(output_path[1], 'w') as score_file:
        json.dump(score, score_file, indent = 4)


if __name__ == '__main__':
    train()
