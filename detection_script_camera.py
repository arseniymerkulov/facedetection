import cv2


from image import Image
from detection_pipeline import DetectionPipeline


camera = cv2.VideoCapture(0)
pipeline = DetectionPipeline()

for i in range(30):
    camera.read()

while True:
    ret, frame = camera.read()

    if not ret:
        continue

    image = Image.preprocess(frame)
    boxes, labels, _ = pipeline.run(image)

    for i in range(len(boxes)):
        sx, sy, ex, ey = Image.convert_box(image, boxes[i])
        cv2.rectangle(image, (sx, sy), (ex, ey), (0, 255, 0), 2)
        cv2.putText(image, labels[i], (ex + 10, ey), 0, 0.3, (0, 0, 0))

    cv2.imshow("image", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
