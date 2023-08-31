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

def gradientBoostingClassifier(X_train, X_test, y_train, y_test):
    # define the model
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

    # fit the model
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    y_pred_1 = model.predict(X_test)
    score = accuracy_score(y_test, y_pred_1)
    # score = roc_auc_score(y_test, y_pred)
    score = model.score(X_test, y_test)

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

def analyse_data(label):
    df_original = pd.read_csv("./data/farmer_survey.csv")
    column_names = df_original.keys()
    df_original, word_to_num = pre_process_string_to_num(df_original)
    df_original = pd.DataFrame(df_original, columns=column_names)

    y = df_original[label].astype('int')
    X = df_original.drop(label, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    gbc_result = gradientBoostingClassifier(X_train, X_test, y_train, y_test)
    print("Accuracy:" + str(gbc_result) + "\n")

    df_synthetic = pd.read_csv("./synthetic_table/farmer_survey_synthetic_with_privacy.csv")
    df_synthetic = pre_process_string_to_num(df_synthetic, word_to_num)[0]
    df_synthetic = pd.DataFrame(df_synthetic, columns=column_names)
    y_test = df_synthetic[label].astype('int')
    X_test = df_synthetic.drop(label, axis=1)

    gbc_result = gradientBoostingClassifier(X_train, X_test, y_train, y_test)
    print("Accuracy:" + str(gbc_result) + "\n")


# analyse_data("livestock_count")
analyse_data("Farm_labour")

