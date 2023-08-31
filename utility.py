import pandas as pd
from datetime import datetime
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
import seaborn as sn
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import csv
import statsmodels.api as sm
from sklearn import tree
from matplotlib import pyplot as plt
import graphviz
features_to_display = 5
from sklearn.utils import resample
import gensim
from gensim.models import word2vec
from sklearn.manifold import TSNE

def logisticRegression(X_train, X_test, y_train, y_test, features):
    # define the model
    model = LogisticRegression()

    # fit the model
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    # score = roc_auc_score(y_test, y_pred)
    score = model.score(X_test, y_test)

    # get importance for class True (1)
    importance = model.coef_
    feature_importance = []
    for i, j in enumerate(importance):
        sorted_index = sorted(range(len(j)), key=j.__getitem__, reverse=True)
        for index in range(features_to_display):
            feature_index = sorted_index[index]
            feature_importance.append(features[feature_index])

    return score

def decisionTree(X_train, X_test, y_train, y_test, features):
    # define the model
    model = DecisionTreeClassifier()

    # fit the model
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    # score = roc_auc_score(y_test, y_pred)
    score = model.score(X_test, y_test)

    # get importance for class True (1)
    importance = model.feature_importances_
    feature_importance = []
    sorted_index = sorted(range(len(importance)), key=importance.__getitem__, reverse=True)
    for index in range(features_to_display):
        feature_index = sorted_index[index]
        feature_importance.append(features[feature_index])

    return score

def randomForestClassifier(X_train, X_test, y_train, y_test, features):
    # define the model
    model = RandomForestClassifier()

    # fit the model
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]

    y_pred_1 = model.predict(X_test)
    score = accuracy_score(y_test, y_pred_1)

    # score = roc_auc_score(y_test, y_pred)
    score = model.score(X_test, y_test)

    # get importance for class True (1)
    importance = model.feature_importances_
    feature_importance = []
    sorted_index = sorted(range(len(importance)), key=importance.__getitem__, reverse=True)
    for index in range(features_to_display):
        feature_index = sorted_index[index]
        feature_importance.append(features[feature_index])

    return score



def gradientBoostingClassifier(X_train, X_test, y_train, y_test, features):
    # define the model
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

    # fit the model
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    y_pred_1 = model.predict(X_test)
    score = accuracy_score(y_test, y_pred_1)
    # score = roc_auc_score(y_test, y_pred)
    score = model.score(X_test, y_test)

    # get importance for class True (1)
    importance = model.feature_importances_
    feature_importance = []
    sorted_index = sorted(range(len(importance)), key=importance.__getitem__, reverse=True)
    for index in range(features_to_display):
        feature_index = sorted_index[index]
        feature_importance.append(features[feature_index])

    return score


def xgbc(X_train, X_test, y_train, y_test, features):
    # define the model
    model = XGBClassifier()

    # fit the model
    model.fit(X_train, y_train)
    #y_pred = model.predict_proba(X_test)[:, 1]
    #score = roc_auc_score(y_test, y_pred)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    y_pred_1 = model.predict(X_test)
    score = accuracy_score(y_test, y_pred_1)

    # get importance for class True (1)
    importance = model.feature_importances_
    feature_importance = []
    sorted_index = sorted(range(len(importance)), key=importance.__getitem__, reverse=True)
    for index in range(features_to_display):
        feature_index = sorted_index[index]
        feature_importance.append(features[feature_index])

    return score


def pre_process_string_to_num(df, word_to_num=None):
    if type(df).__module__ != np.__name__:
        df = df.fillna('')
        df = df.to_numpy()

    # converting strings
    if word_to_num is None:
        word_to_num = {}

    count = np.empty(shape=df.shape[1], dtype=int)
    for s in range(count.shape[0]):
        count[s] = 0

    for i in range(0, df.shape[0]):
        for j in range(df.shape[1]):
            try:
                df[i, j] = float(df[i, j])
            except:
                key = (j, df[i, j])
                if key not in word_to_num:
                    word_to_num[key] = count[j]
                    count[j] = count[j] + 1
                df[i, j] = float(word_to_num[key])

    return df, word_to_num

def analyse_data(csv_name, label):
    df = pd.read_csv(csv_name)

    column_names = df.keys()
    print(column_names)
    df, word_to_num = pre_process_string_to_num(df)
    df = pd.DataFrame(df, columns=column_names)

    y = df[label].astype('int')
    X = df.drop(label, axis=1)
    features = X.keys()
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print(X_train.shape)

    ML_models = ["logisticRegression", "decisionTree", "RandomForestClassifier", "GradientBoostingClassifier", "XGBoost"]

    print("Original")

    lr_result = logisticRegression(X_train, X_test, y_train, y_test, features)
    print(ML_models[0] + ":" + str(lr_result) + "\n")

    dt_result = decisionTree(X_train, X_test, y_train, y_test, features)
    print(ML_models[1] + ":" + str(dt_result) + "\n")

    rfc_result = randomForestClassifier(X_train, X_test, y_train, y_test, features)
    print(ML_models[2] + ":" + str(rfc_result) + "\n")

    gbc_result = gradientBoostingClassifier(X_train, X_test, y_train, y_test, features)
    print(ML_models[3] + ":" + str(gbc_result) + "\n")

    """xgbc_result = xgbc(X_train, X_test, y_train, y_test, features)
    print(ML_models[4] + ":" + str(xgbc_result) + "\n")"""

    df_new = pd.read_csv("./synthetic_table/farmer_survey_synthetic_without_privacy.csv")
    column_names = df.keys()
    df_new = pre_process_string_to_num(df_new, word_to_num)[0]
    df_new = pd.DataFrame(df_new, columns=column_names)
    y_train = df_new[label].astype('int')
    X_train = df_new.drop(label, axis=1)

    print("Synthetic")
    lr_result = logisticRegression(X_train, X_test, y_train, y_test, features)
    print(ML_models[0] + ":" + str(lr_result) + "\n")

    dt_result = decisionTree(X_train, X_test, y_train, y_test, features)
    print(ML_models[1] + ":" + str(dt_result) + "\n")

    rfc_result = randomForestClassifier(X_train, X_test, y_train, y_test, features)
    print(ML_models[2] + ":" + str(rfc_result) + "\n")

    gbc_result = gradientBoostingClassifier(X_train, X_test, y_train, y_test, features)
    print(ML_models[3] + ":" + str(gbc_result) + "\n")

    """xgbc_result = xgbc(X_train, X_test, y_train, y_test, features)
    print(ML_models[4] + ":" + str(xgbc_result) + "\n")"""



analyse_data("./data/farmer_survey.csv", "household_type")
