import cv2
import glob
import os


import hyperparams
from image import Image
from detection_pipeline import DetectionPipeline


pipeline = DetectionPipeline()

total = 0
success = 0

for directory in os.listdir(hyperparams.simple_images_path):
    images = glob.glob(f'{hyperparams.simple_images_path}/{directory}/*.jpg')

    for image_path in images:
        image = Image.load(image_path)

        if image is None:
            continue

        boxes, labels, _ = pipeline.run(image)

        for i in range(len(boxes)):
            #  sx, sy, ex, ey = Image.convert_box(image, boxes[i])
            # cv2.rectangle(image, (sx, sy), (ex, ey), (0, 255, 0), 2)
            # cv2.putText(image, labels[i], (ex + 10, ey), 0, 0.3, (255, 0, 0))

            print(directory)
            print(labels[i])

            total += 1
            success += 1 if directory == labels[i] else 0

        # cv2.imshow("image", image)
        # cv2.waitKey()

print(f'total: {total}')
print(success / total)
