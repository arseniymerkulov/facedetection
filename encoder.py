import dlib
import numpy as np


import hyperparams


class Encoder:
    def __init__(self, path=hyperparams.model_face_recognition_path):
        self.encoder = dlib.face_recognition_model_v1(path)

    def encode(self,
               image,
               landmarks68=None,
               face_chip=None,
               jitters=hyperparams.encoder_jitters,
               padding=hyperparams.encoder_padding):
        if landmarks68 is not None:
            output = self.encoder.compute_face_descriptor(image, landmarks68, jitters, padding)
        elif face_chip is not None:
            output = self.encoder.compute_face_descriptor(face_chip)
        else:
            output = [0]

        return np.array(output)
