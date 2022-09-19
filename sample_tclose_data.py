from scipy.stats import wasserstein_distance

import logging
import numpy as np
import pandas as pd
import ctgan
import os


class SampleTCloseData:
    def __init__(self, t_threshold, generator_model):
        self.t_threshold = t_threshold
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

        np1 = (np1 - np1.min(axis=0)) / (np1.max(axis=0) - np1.min(axis=0))
        np2 = (np2 - np2.min(axis=0)) / (np2.max(axis=0) - np2.min(axis=0))

        distances = []
        for i in range(1, np1.shape[1]):
            distances.append(wasserstein_distance((np.asarray(np1[:, i])), (np.asarray(np2[:, i]))))

        return max(distances)

    def sample_synthetic_table(self, original_data, initial_sample_size=100, sampling_step=10):
        emd_max = float('inf')
        first_step = True

        column_names = original_data.keys()
        original_data, word_to_num = self.pre_process_string_to_num(original_data)

        while emd_max > self.t_threshold:
            if first_step:
                synthetic_data = self.generator_model.sample(n=initial_sample_size)
                first_step = False
            else:
                new_samples = self.generator_model.sample(n=sampling_step)
                synthetic_data.append(new_samples)

            synthetic_data, word_to_num = self.pre_process_string_to_num(synthetic_data, word_to_num)

            emd_max = self.get_emd_distance(np.array(original_data), np.array(synthetic_data))

        synthetic_df = pd.DataFrame(synthetic_data, columns=column_names)
        try:
            if not os.path.exists("synthetic_table"):
                os.mkdir("synthetic_table")
            synthetic_df.to_csv("synthetic_table/synthetic_data.csv", index=False)

            logging.info("Synthetic Table successfully saved")
        except Exception as e:
            print(e)
            logging.error("Synthetic Table could not be saved")

        return synthetic_data

