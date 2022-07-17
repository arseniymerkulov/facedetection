import dlib


import hyperparams


class ShapePredictor:
    def __init__(self, path=hyperparams.model_shape_predictor_5_path):
        self.shape_predictor = dlib.shape_predictor(path)

    def get_landmarks(self, image, box):
        return self.shape_predictor(image, box)
