import glob
import os
import cv2


import hyperparams
from image import Image


def augment_dataset():
    for directory in os.listdir(hyperparams.classifier_dataset_image_path):
        images_directory = f'{hyperparams.classifier_dataset_image_path}/{directory}'
        images = glob.glob(f'{images_directory}/*.jpg')

        for image_path in images:
            image_filename, image_extension = os.path.splitext(os.path.basename(image_path))
            image = Image.load(image_path)

            image_flipped = cv2.flip(image, 1)
            cv2.imwrite(f'{images_directory}/{image_filename}_flipped.{image_extension}',
                        cv2.cvtColor(image_flipped, cv2.COLOR_RGB2BGR))


def clear_dataset():
    for directory in os.listdir(hyperparams.classifier_dataset_image_path):
        images_directory = f'{hyperparams.classifier_dataset_image_path}/{directory}'
        images = glob.glob(f'{images_directory}/*.jpg')

        for image_path in images:
            image_filename = os.path.splitext(os.path.basename(image_path))[0]

            if '_flipped' in image_filename:
                os.unlink(image_path)


augment_dataset()
# clear_dataset()



