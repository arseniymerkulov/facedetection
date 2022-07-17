import cv2
import dlib


import hyperparams


class Image:
    @staticmethod
    def preprocess(image, image_width=hyperparams.image_width, image_height=None):
        if image_height is None:
            ratio = image.shape[0] / image.shape[1]
            image_height = int(image_width * ratio)

        image = cv2.resize(image, (image_width, image_height))

        return image

    @staticmethod
    def load(path, image_width=hyperparams.image_width, image_height=None):
        image = Image.preprocess(cv2.imread(path), image_width, image_height)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    @staticmethod
    def convert_box(image, box):
        sx = box.left()
        sy = box.top()
        ex = box.right()
        ey = box.bottom()

        sx = max(0, sx)
        sy = max(0, sy)
        ex = min(ex, image.shape[1])
        ey = min(ey, image.shape[0])

        return sx, sy, ex, ey

    @staticmethod
    def convert_boxes(image, boxes):
        return [Image.convert_box(image, box) for box in boxes]

    @staticmethod
    def get_default_box(image):
        h, w, _ = image.shape
        return dlib.rectangle(0, h, w, 0)
