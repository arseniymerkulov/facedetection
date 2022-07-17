import cv2
import glob


import hyperparams
from final_form.image import Image
from detection_pipeline import DetectionPipeline


pipeline = DetectionPipeline()
images = glob.glob(f'{hyperparams.group_images_path}/*.jpg')

for image_path in images:
    image = Image.load(image_path)
    boxes, labels, _ = pipeline.run(image)

    for i in range(len(boxes)):
        sx, sy, ex, ey = Image.convert_box(image, boxes[i])
        cv2.rectangle(image, (sx, sy), (ex, ey), (0, 255, 0), 2)
        cv2.putText(image, labels[i], (ex + 10, ey), 0, 0.3, (255, 0, 0))

    cv2.imshow("image", image)
    cv2.waitKey()
