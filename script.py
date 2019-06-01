import utils
import preprocess as pre
import feature_extraction as fe
import XGBoost as xgb
from sklearn.model_selection import train_test_split
import performances as per
import visualization as vis
import numpy as np
import Baseline as bl
from explanation import explain_model
import os
import pathlib

path_object = pathlib.Path('outputs')
if not path_object.exists():
    os.makedirs('outputs')
# get tagged df
tagged_df = utils.read_to_df()  # Vigo data

# pre process
tagged_df = pre.preprocess(tagged_df)

# extract features
feature_list = ['post_length', 'tfidf', 'topics', 'screamer', 'words', 'off_dis', 'not_off_dis']
X = fe.extract_features(tagged_df, feature_list)
y = (tagged_df['cb_level'] == 3).astype(int)
X = X.drop(columns=['id'])

# split data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

performances_list = {}
auc_list = {}

# 1.baseline
y_pred_bl = bl.run_baseline(tagged_df)
performances_bl = per.get_performances(y, y_pred_bl)
performances_list['baseline'] = performances_bl

# 2.XGBoost
xgb_obj = xgb.XGBoost()
xgb_classifier = xgb_obj.train(X_train, y_train)
y_prob_xgb = xgb_obj.predict(X_test)
y_pred_xgb = np.where(y_prob_xgb > 0.5, 1, 0)
performances_xgb = per.get_performances(y_test, y_pred_xgb)
performances_list['XGBoost'] = performances_xgb

# visualization
roc_auc_bl, fpr_bl, tpr_bl = per.get_roc_auc(y, y_pred_bl)
auc_list['baseline'] = roc_auc_bl
roc_auc_xgb, fpr_xgb, tpr_xgb = per.get_roc_auc(y_test, y_prob_xgb)
auc_list['XGBoost'] = roc_auc_xgb

vis.plot_roc_curve(roc_auc_bl, fpr_bl, tpr_bl,'baseline')
vis.plot_roc_curve(roc_auc_xgb, fpr_xgb, tpr_xgb, 'xgboost')
vis.plot_models_compare(performances_bl, performances_xgb)

# SHAP for XGBoost:
path_object = pathlib.Path('pictures')
if not path_object.exists():
    os.makedirs('pictures')
explain_model(xgb_obj.get_booster(), X_test, True)

acc_bl = per.get_accuracy(y, y_pred_bl)
acc_xgb = per.get_accuracy(y_test, y_pred_xgb)

print('accuracy for baseline: ', acc_bl)
print('accuracy for xgboost: ', acc_xgb)
