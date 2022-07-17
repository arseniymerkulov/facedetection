import dlib
import cv2
import glob
import os
import json
import numpy as np


import hyperparams
from image import Image
from detection_pipeline import DetectionPipeline


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


pipeline = DetectionPipeline()


for directory in os.listdir(hyperparams.classifier_dataset_image_path):
    images = glob.glob(f'{hyperparams.classifier_dataset_image_path}/{directory}/*.jpg')

    for image_path in images:
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        image = Image.load(image_path,
                           image_width=hyperparams.face_image_width,
                           image_height=hyperparams.face_image_height)

        boxes, labels, features = pipeline.run(image)
        print(features)

        if len(features) > 0:
            with open(f'{hyperparams.classifier_dataset_json_path}/{directory}/{file_name}.json', 'w') as file:
                file.write(json.dumps({"features": features[0]}, cls=NumpyEncoder))