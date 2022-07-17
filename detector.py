import dlib


import hyperparams


class Detector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect(self, image, upsample=hyperparams.detector_upsample):
        return self.detector(image, upsample)
