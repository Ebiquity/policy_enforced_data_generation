import sys
from ctgan import CTGANSynthesizer
import pandas as pd
from generator_model import GeneratorModel
from sample_tclose_data import SampleTCloseData


def table_generator(original_data_path, ):
    original_data = pd.read_csv(original_data_path)
    # generator_model = GeneratorModel(original_data).train_and_save_ctgan()
    ctgan = CTGANSynthesizer()
    generator_model = ctgan.load("./trained_model/trained_generator.pth")
    sampler = SampleTCloseData(generator_model=generator_model)
    synthetic_table = sampler.sample_synthetic_table(original_data)

    return synthetic_table


table_generator(original_data_path="./data/farmer_survey.csv")
