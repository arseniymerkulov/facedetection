import cv2
import glob
import os


import hyperparams
from image import Image
from detection_pipeline import DetectionPipeline


pipeline = DetectionPipeline()

image = 'data/group_face_dataset/IKhajcTRV7Q.jpg'
image = Image.load(image)

boxes, labels, scores = pipeline.run(image)

for i in range(len(boxes)):
    sx, sy, ex, ey = Image.convert_box(image, boxes[i])
    cv2.rectangle(image, (sx, sy), (ex, ey), (0, 255, 0), 2)
    cv2.putText(image, f'{labels[i]} {round(100 * scores[i], 1)}%', (ex + 10, ey), 0, 0.5, (0, 255, 0), 2)

cv2.imwrite('output.jpg', image)
# cv2.imshow("image", image)
# cv2.waitKey()

