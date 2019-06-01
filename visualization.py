import matplotlib.pyplot as plt
import utils
import os
import pandas as pd
import wordcloud
import xgboost as xgb
import numpy as np


def plot_tf_idf_post(dictionary_tf_idf, title):
    dic_post = dict(dictionary_tf_idf[title])
    dic_post_travers = {}
    for term,val in dic_post.items():
        dic_post_travers[utils.traverse(term)] = val
    df2 = pd.DataFrame.from_dict(dic_post_travers,orient='index').sort_values(by=0, ascending=False)
    pl = df2.plot(kind='bar', figsize=(15,7), fontsize=8, legend=False, title=utils.traverse(title))
    for p in pl.patches:
        pl.annotate(str(p.get_height()), (p.get_x() * 0.98, p.get_height() * 1.001), fontsize=14)
    plt.show()


def plot_length_posts(dictionary_length, title):
    df2 = pd.DataFrame.from_dict(dictionary_length, orient='index').sort_values(by=0, ascending=False)
    pl = df2.plot(kind='bar', figsize=(15, 7), fontsize=8, legend=False, title=utils.traverse(title))
    for p in pl.patches:
        pl.annotate(str(p.get_height()), (p.get_x() * 0.98, p.get_height() * 1.001), fontsize=14)
    plt.show()


def print_tf_idf_dict(tf_idf_dict):
    for key, value in tf_idf_dict.items():
        print('post: ')
        print(key)
        for v in value:
            print('word: ' + str(v[0]) + ', tf-idf: ' + str(v[1]))


def plot_part_of_day(dictionary_time, title):
    df2 = pd.DataFrame.from_dict(dictionary_time, orient='index').sort_values(by=0, ascending=False)
    pl = df2.plot(kind='bar', figsize=(15, 7), fontsize=8, legend=False, title=utils.traverse(title))
    for p in pl.patches:
        pl.annotate(str(p.get_height()), (p.get_x() * 0.98, p.get_height() * 1.001), fontsize=14)
    plt.show()


def plot_roc_curve(roc_auc, fpr, tpr, name):
    plt.figure()
    lw = 1
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic -'+str(name))
    plt.legend(loc="lower right")
    plt.show()


def plot_feature_importance_xgb(booster):
    xgb.plot_importance(booster, importance_type='gain')
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.show()


def plot_models_compare(per1, per2):
    fig, ax = plt.subplots()
    index = np.arange(3)
    ax.bar(index, [per1[key] for key in sorted(per1.keys())], color=(0.5, 0.4, 0.8, 0.4), width=0.2, label='Baseline')
    ax.bar(index+0.25, [per2[key] for key in sorted(per2.keys())], color=(0.8, 0.5, 0.4, 0.6), width=0.2, label='XGBoost')
    ax.set_xlabel('Performances')
    ax.set_ylabel('')
    ax.set_title('Model compare')
    ax.set_xticks(index+0.3)
    ax.set_xticklabels(['F-Measure', 'Precision', 'Recall'])
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_precision_recall_curve(p, r):
    plt.step(r, p, color='b', alpha=0.2, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve')
    plt.legend(loc="lower right")

