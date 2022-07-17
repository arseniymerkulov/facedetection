import dlib


from detector import Detector
from shape_predictor import ShapePredictor
from aligner import Aligner
from encoder import Encoder
from classifier import Classifier


# todo: get 5-landmarks predictor
# 5-landmarks for aligning and 68-landmarks for encoding


class DetectionPipeline:
    def __init__(self):
        self.detector = Detector()
        self.shape_predictor_5 = ShapePredictor()
        self.encoder = Encoder()
        self.classifier = Classifier()

        self.aligner68 = Aligner((42, 47),
                                 (36, 41))
        self.aligner5 = Aligner((0, 1),
                                (2, 3))

    def run(self, image):
        boxes = self.detector.detect(image)
        labels = []
        features = []

        for box in boxes:
            shape = self.shape_predictor_5.get_landmarks(image, box)
            face_chip = dlib.get_face_chip(image, shape)
            encoding = self.encoder.encode(image, face_chip=face_chip)

            features.append(encoding)
            labels.append(self.classifier.classify(encoding))

            # face = self.aligner5.align(image, box)
            # landmarks68 = self.aligner68.get_landmarks(face, Detector.get_default_box(face))
            # features = self.encoder.encode(face, landmarks68)

        return boxes, labels, features




