from ctgan import CTGANSynthesizer
import configparser
import ast
import os
import logging


class GeneratorModel:
    def __init__(self, data):
        self.data = data

        config = configparser.ConfigParser()
        config.read("attributes.ini")
        self.discrete_attributes = ast.literal_eval(config.get("ATTRIBUTES", "discrete_attributes"))
        print(self.discrete_attributes)

        if len(self.discrete_attributes) == 0:
            logging.warning("No Discrete Attribute defined")

    def train_and_save_ctgan(self, epochs=300):
        ctgan = CTGANSynthesizer(epochs=epochs)
        ctgan.fit(self.data, self.discrete_attributes)
        logging.info("CTGAN model successfully trained")

        try:
            if not os.path.exists("trained_model"):
                os.mkdir("trained_model")
            ctgan.save("trained_model/generator_model.pth")

            logging.info("Generator model successfully saved")
        except:
            logging.error("Generator model could not be saved")

        return ctgan
