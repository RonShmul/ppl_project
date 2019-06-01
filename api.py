import preprocess as pre
import feature_extraction as fe
import performances as per
import utils
import explanation as exp
import XGBoost as xgb
import numpy as np
import pandas as pd
import shutil
import os
import pathlib

HERE = pathlib.Path(__file__).parent
FEATURE_LIST = ['post_length', 'tfidf', 'topics', 'screamer', 'words', 'off_dis', 'not_off_dis']


def train_file(file_path):
    path_object = pathlib.Path(HERE / 'outputs')
    if path_object.exists():
        shutil.rmtree(HERE / 'outputs')
        os.makedirs(HERE / 'outputs')
    tagged_df = utils.read_to_df(file_path)
    tagged_df = pre.preprocess(tagged_df)
    X = fe.extract_features(tagged_df, FEATURE_LIST)
    y = (tagged_df['cb_level'] == 3).astype(int)
    X = X.drop(columns=['id'])
    xgb_obj = xgb.XGBoost()
    xgb_obj.train(X, y)
    exp.explain_model(xgb_obj.model, X)
    utils.save_model(xgb_obj.model, os.path.join(HERE / 'outputs', 'XGBoost.pkl'))


def predict(post, explainability=True):
    if len(os.listdir(HERE / 'outputs')) == 0:
        return {'error': "Please train the model with train data set first.."}

    model = utils.get_model(os.path.join(HERE / 'outputs', 'XGBoost.pkl'))
    xgb_obj = xgb.XGBoost()
    xgb_obj.model = model
    post_dataframe = pd.DataFrame({'id': [1], 'text': [post]})
    post_dataframe = pre.preprocess(post_dataframe)
    X = fe.extract_features(post_dataframe, FEATURE_LIST)
    X = X.drop(columns=['id'])
    y_prob = xgb_obj.predict(X)
    pred = np.where(y_prob > 0.5, 1, 0)
    result = {'class': int(pred[0])}
    if explainability:
        result['explain'] = exp.explain_class(model, X)
    return result


def get_performances(file_path):
    model = utils.get_model(HERE / 'outputs/XGBoost.pkl')
    xgb_obj = xgb.XGBoost()
    xgb_obj.model = model
    df = utils.read_to_df(file_path)
    df = pre.preprocess(df)
    X = fe.extract_features(df, FEATURE_LIST)
    X = X.drop(columns=['id'])
    y = (df['cb_level'] == 3).astype(int)
    y_prob_rf = xgb_obj.predict(X)
    pred = np.where(y_prob_rf > 0.5, 1, 0)
    return per.get_performances(y, pred)
