import dlib
import numpy as np


import hyperparams


class Encoder:
    def __init__(self, path=hyperparams.model_face_recognition_path):
        self.encoder = dlib.face_recognition_model_v1(path)

    def encode(self, face_chip):
        output = self.encoder.compute_face_descriptor(face_chip)
        return np.array(output)
