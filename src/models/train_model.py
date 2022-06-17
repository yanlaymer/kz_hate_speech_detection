import click
import pandas as pd
import lightgbm as lgb
import joblib as jb
import mlflow
from mlflow.models.signature import infer_signature

import os
from dotenv import load_dotenv

from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import json

mlflow.set_tracking_uri("http://localhost:5000")


@click.command()
@click.argument("input_paths", type=click.Path(exists=True), nargs=2)
@click.argument("output_path", type=click.Path(), nargs=2)
def train(input_paths: List[str], output_path: List[str]):
    with mlflow.start_run():
        train_df = pd.read_csv(input_paths[0])
        test_df = pd.read_csv(input_paths[1])

        train_df.dropna(subset='text')
        test_df.dropna(subset='text')

        X_train = train_df['text']
        y_train = train_df['metka']
        X_holdout = test_df['text']
        y_holdout = test_df['metka']
        print(X_train)
        print(X_train.shape, X_holdout.shape, y_train.shape, y_holdout.shape)
        w2v = TfidfVectorizer(max_features=600)
        X_train = w2v.fit_transform(X_train.values.astype('U'))
        X_holdout = w2v.transform(X_holdout.values.astype('U'))

        print(X_train.shape, X_holdout.shape, y_train.shape, y_holdout.shape)

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_holdout, y_holdout, reference=lgb_train)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'verbose': 1,
            'max_depth': 11,
            'num_leaves': 150,
            'learning_rate': 0.25,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'n_estimators': 1000,
        }

        # gbm = lgb.train(params,
        #                 lgb_train,
        #                 num_boost_round=200,
        #                 valid_sets=lgb_eval,
        #                 verbose_eval=False,
        #                 early_stopping_rounds=30)  # categorical_feature=['building_type']
        gbm = lgb.LGBMClassifier(**params)
        gbm.fit(X_train, y_train)
        jb.dump(gbm, output_path[0])

        y_predicted = gbm.predict(X_holdout)
        score = dict(
            accuracy=accuracy_score(y_holdout, y_predicted),
            precision=precision_score(y_holdout, y_predicted),
            recall=recall_score(y_holdout, y_predicted),
            f1=f1_score(y_holdout, y_predicted)
        )

        with open(output_path[1], 'w') as score_file:
            json.dump(score, score_file, indent=4)

        signature = infer_signature(X_holdout, y_predicted)
        mlflow.log_params(params)
        mlflow.log_metrics(score)
        mlflow.lightgbm.log_model(lgb_model=gbm,
                                  artifact_path="model",
                                  registered_model_name="hate_sp_lgbm",
                                  signature=signature)


if __name__ == '__main__':
    train()
