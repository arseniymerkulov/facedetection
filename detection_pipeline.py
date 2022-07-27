import dlib


import hyperparams
from detector import Detector
from shape_predictor import ShapePredictor
from encoder import Encoder
from classifier import Classifier


# todo: get 5-landmarks predictor
# 5-landmarks for aligning and 68-landmarks for encoding


class DetectionPipeline:
    def __init__(self):
        self.detector = Detector()
        self.shape_predictor_5 = ShapePredictor()
        self.shape_predictor_68 = ShapePredictor(hyperparams.model_shape_predictor_68_path)
        self.encoder = Encoder()
        self.classifier = Classifier()

    def run(self, image):
        boxes = self.detector.detect(image)
        labels = []
        scores = []

        for box in boxes:
            shape = self.shape_predictor_68.get_landmarks(image, box)
            face_chip = dlib.get_face_chip(image, shape)
            encoding = self.encoder.encode(face_chip)
            label, score = self.classifier.classify(encoding)
            labels.append(label)
            scores.append(score)

        return boxes, labels, scores




