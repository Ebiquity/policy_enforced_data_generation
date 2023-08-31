from scipy.stats import wasserstein_distance

import logging
import numpy as np
import pandas as pd
import ctgan
import os
from sklearn.preprocessing import normalize


class SampleTCloseData:
    def __init__(self, generator_model):
        self.generator_model = generator_model

    def pre_process_string_to_num(self, df, word_to_num=None):
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
                    df[i, j] = word_to_num[key]

        return df, word_to_num

    def get_emd_distance(self, np1, np2):
        if np1.shape[1] != np2.shape[1]:
            logging.error("Different number of attributes")
            raise ValueError

        np1 = normalize(np1, axis=1, norm='l1')
        np2 = normalize(np2, axis=1, norm='l1')

        """if not (np1.max(axis=0) - np1.min(axis=0)) == 0:
            np1 = (np1 - np1.min(axis=0)) / (np1.max(axis=0) - np1.min(axis=0))

        if not (np2.max(axis=0) - np2.min(axis=0)) == 0:
            np2 = (np2 - np2.min(axis=0)) / (np2.max(axis=0) - np2.min(axis=0))"""

        distances = []
        for i in range(np1.shape[1]):
            distances.append(wasserstein_distance((np.asarray(np1[:, i])), (np.asarray(np2[:, i]))))

        return distances


    def get_emd_threshold(self, column_name):
        df = pd.read_csv("./data/farmer_survey_metadata.csv")
        try:
            privacy_level = df[df["name"] == column_name].iloc[0][1]
        except IndexError as e:
            return 0.5

        # print(privacy_level)
        if pd.isnull(privacy_level):
            return 0.5
        elif privacy_level == "Low" or "low" or "Loe":
            return 0.5
        elif privacy_level == "Med":
            return 0.2
        elif privacy_level == "High":
            return 0.1

    def is_privacy_satisfied(self, column_names, emd_distance):
        for i, column_name in enumerate(column_names):
            # print(column_name)
            threshold = self.get_emd_threshold(column_name)
            # print(threshold)
            if emd_distance[i] > threshold:
                # print(column_name)
                return False

        return True

    def sample_synthetic_table(self, original_data, initial_sample_size=1000, sampling_step=10):
        emd_max = float('inf')
        first_step = True

        column_names = original_data.keys()
        original_data, word_to_num = self.pre_process_string_to_num(original_data)

        privacy_constrained = False
        privacy_satisfied = False

        while not privacy_satisfied:
            if first_step:
                synthetic_data = self.generator_model.sample(n=initial_sample_size)
                first_step = False
            else:
                new_samples = self.generator_model.sample(n=sampling_step)
                synthetic_data.append(new_samples)

            synthetic_data_num, word_to_num = self.pre_process_string_to_num(synthetic_data, word_to_num)

            emd_distance = self.get_emd_distance(np.array(original_data), np.array(synthetic_data_num))

            print("columns: " + str(len(column_names)))
            print("distances: " + str(len(emd_distance)))
            if privacy_constrained:
                privacy_satisfied = self.is_privacy_satisfied(column_names, emd_distance)
            else:
                max_emd_distance = max(emd_distance)
                if max_emd_distance < 0.1:
                    privacy_satisfied = True

        synthetic_df = pd.DataFrame(synthetic_data, columns=column_names)
        try:
            if not os.path.exists("synthetic_table"):
                os.mkdir("synthetic_table")
            synthetic_df.to_csv("synthetic_table/farmer_survey_synthetic_without_privacy.csv", index=False)

            logging.info("Synthetic Table successfully saved")
        except Exception as e:
            print(e)
            logging.error("Synthetic Table could not be saved")

        return synthetic_data

