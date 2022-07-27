import glob
import os
import json
import numpy as np
import cv2


import hyperparams
from image import Image
from encoder import Encoder


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


encoder = Encoder()

if not os.path.exists(hyperparams.classifier_dataset_json_path):
    os.mkdir(hyperparams.classifier_dataset_json_path)

for directory in os.listdir(hyperparams.classifier_dataset_image_path):
    images = glob.glob(f'{hyperparams.classifier_dataset_image_path}/{directory}/*.jpg')
    json_directory_path = f'{hyperparams.classifier_dataset_json_path}/{directory}'

    if not os.path.exists(json_directory_path):
        os.mkdir(json_directory_path)

    for image_path in images:
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        image = Image.load(image_path,
                           image_width=hyperparams.face_image_width,
                           image_height=hyperparams.face_image_height)

        if image is None:
            continue

        features = encoder.encode(image)
        print(features)

        if len(features) > 0:
            with open(f'{json_directory_path}/{file_name}.json', 'w') as file:
                file.write(json.dumps(features, cls=NumpyEncoder))
