import tensorflow as tf
import numpy as np
import json
import glob
import os


import hyperparams


class Classifier:
    def __init__(self, path=hyperparams.model_face_classification_path):
        self.model = Classifier.get_model_architecture()
        self.model.load_weights(path)

    @staticmethod
    def get_model_architecture():
        return tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(128,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2, activation='sigmoid')
        ])

    @staticmethod
    def get_json_dataset():
        dataset = []
        labels = []

        for directory in os.listdir(hyperparams.classifier_dataset_json_path):
            images = glob.glob(f'{hyperparams.classifier_dataset_json_path}/{directory}/*.json')

            for image_path in images:
                with open(image_path, 'rb') as file:
                    image = json.loads(file.read())

                image = np.array(image['features'])
                dataset.append(image)
                labels.append([1., 0.] if directory == hyperparams.classifier_categories[0] else [0., 1.])

        return np.array(dataset), np.array(labels)

    def classify(self, features):
        output = self.model(np.expand_dims(features, axis=0))
        return hyperparams.classifier_categories[np.argmax(output)]
