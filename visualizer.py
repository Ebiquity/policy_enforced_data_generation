import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns


def stuff():
    df = pd.DataFrame()
    df["comp-1"] = df_original_embedded[:, 0]
    df["comp-2"] = df_original_embedded[:, 1]
    sns_plot = sns.scatterplot(x="comp-1", y="comp-2",
                               palette=sns.color_palette("hls", 3),
                               data=df, label="Original Data")
    plt.savefig("./output/tsne_visualization/original.png")

    df_1 = pd.DataFrame()
    df_1["comp-1"] = df_without_policy_embedded[:, 0]
    df_1["comp-2"] = df_without_policy_embedded[:, 1]
    sns_plot = sns.scatterplot(x="comp-1", y="comp-2",
                               palette=sns.color_palette("hls", 3),
                               data=df_1, label="Synthetic Data").set(
        title="Comparison of T-SNE projection without Policy enforcement")
    plt.savefig("./output/tsne_visualization/without_policy.png")
    plt.close()

def stuff_2():
    df = pd.DataFrame()
    df["comp-1"] = df_original_embedded[:, 0]
    df["comp-2"] = df_original_embedded[:, 1]
    sns_plot = sns.scatterplot(x="comp-1", y="comp-2",
                               palette=sns.color_palette("hls", 3),
                               data=df, label="Original Data")
    # plt.savefig("./output/tsne_visualization/original.png")

    df_2 = pd.DataFrame()
    df_2["comp-1"] = df_with_policy_embedded[:, 0]
    df_2["comp-2"] = df_with_policy_embedded[:, 1]

    sns_plot = sns.scatterplot(x="comp-1", y="comp-2",
                               palette=sns.color_palette("hls", 3),
                               data=df_2, label="Synthetic Data").set(
        title="Comparison of T-SNE projection with Policy enforcement")

    plt.savefig("./output/tsne_visualization/with_policy.png")

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

df_original = pd.read_csv("./data/farmer_survey.csv")
df_without_policy = pd.read_csv("./synthetic_table/farmer_survey_synthetic_without_privacy.csv")
df_with_policy = pd.read_csv("./synthetic_table/farmer_survey_synthetic_with_privacy.csv")

df_original, word_to_num = pre_process_string_to_num(df_original)
df_without_policy = pre_process_string_to_num(df_without_policy, word_to_num)[0]
df_with_policy = pre_process_string_to_num(df_with_policy, word_to_num)[0]


df_original_embedded = TSNE(n_components=2, init='random', perplexity=3).fit_transform(df_original)
df_without_policy_embedded = TSNE(n_components=2, init='random', perplexity=3).fit_transform(df_without_policy)
df_with_policy_embedded = TSNE(n_components=2, init='random', perplexity=3).fit_transform(df_with_policy)

stuff()
stuff_2()