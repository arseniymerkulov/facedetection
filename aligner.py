import numpy as np
import cv2


import hyperparams


class Aligner:
    def __init__(self,
                 left_eye_landmarks,
                 right_eye_landmarks,
                 desired_left_eye=(0.35, 0.35),
                 desired_face_width=hyperparams.face_image_width,
                 desired_face_height=hyperparams.face_image_height):

        self.left_eye_landmarks = left_eye_landmarks
        self.right_eye_landmarks = right_eye_landmarks

        self.desired_left_eye = desired_left_eye
        self.desired_face_width = desired_face_width
        self.desired_face_height = desired_face_height

        if self.desired_face_height is None:
            self.desired_face_height = self.desired_face_width

    @staticmethod
    def _landmarks_to_np(landmarks):
        coords = np.zeros((landmarks.num_parts, 2), dtype="int")

        for i in range(0, landmarks.num_parts):
            coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

        return coords

    def align(self, image, landmarks):
        landmarks = Aligner._landmarks_to_np(landmarks)

        (le_start, le_end) = self.left_eye_landmarks
        (re_start, re_end) = self.right_eye_landmarks

        le_landmarks = landmarks[le_start:le_end]
        re_landmarks = landmarks[re_start:re_end]

        le_center = le_landmarks.mean(axis=0).astype("int")
        re_center = re_landmarks.mean(axis=0).astype("int")

        dx = re_center[0] - le_center[0]
        dy = re_center[1] - le_center[1]
        angle = np.degrees(np.arctan2(dy, dx)) - 180

        desired_re_x = 1.0 - self.desired_left_eye[0]
        desired_dist = desired_re_x - self.desired_left_eye[0]
        desired_dist *= self.desired_face_width

        dist = np.sqrt((dx ** 2) + (dy ** 2))
        scale = desired_dist / dist

        center = ((le_center[0] + re_center[0]) // 2,
                  (le_center[1] + re_center[1]) // 2)

        rot_matrix = cv2.getRotationMatrix2D((int(center[0]), int(center[1])), angle, scale)

        tx = self.desired_face_width * 0.5
        ty = self.desired_face_height * self.desired_left_eye[1]

        rot_matrix[0, 2] += (tx - center[0])
        rot_matrix[1, 2] += (ty - center[1])

        output = cv2.warpAffine(image,
                                rot_matrix,
                                (self.desired_face_width, self.desired_face_height),
                                flags=cv2.INTER_CUBIC)

        return output
