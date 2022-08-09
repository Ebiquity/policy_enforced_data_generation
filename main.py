import sys
import pandas as pd
from generator_model import GeneratorModel
from sample_tclose_data import SampleTCloseData


def table_generator(original_data_path, t_threshold):
    original_data = pd.read_csv(original_data_path)
    generator_model = GeneratorModel(original_data).train_and_save_ctgan()
    sampler = SampleTCloseData(t_threshold=t_threshold, generator_model=generator_model)
    synthetic_table = sampler.sample_synthetic_table(original_data)

    return synthetic_table


if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 0:
        print("No Input Data provided")
        exit()
    elif len(args) == 1:
        original_data_path = args[0]
        t_threshold = 0.6
    else:
        original_data_path = args[0]
        t_threshold = args[1]

    table_generator(original_data_path=original_data_path, t_threshold=t_threshold)
