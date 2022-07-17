model_shape_predictor_5_path = 'models/shape_predictor_68_face_landmarks.dat'
model_shape_predictor_68_path = 'models/shape_predictor_5_face_landmarks.dat'
model_face_recognition_path = 'models/dlib_face_recognition_resnet_model_v1.dat'
model_face_classification_path = 'models/classifier/checkpoint'

group_images_path = 'data/group_images'

image_width = 1024

face_image_width = 180
face_image_height = 180

detector_upsample = 1

encoder_jitters = 1
encoder_padding = 0.25

classifier_dataset_image_path = 'data/image_dataset'
classifier_dataset_json_path = 'data/json_dataset'

classifier_batch_size = 4
classifier_val_split = .2
classifier_epochs = 500
classifier_learning_rate = 0.0001
classifier_categories = ['arseny', 'other']
