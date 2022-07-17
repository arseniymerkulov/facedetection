import numpy as np


import hyperparams
from classifier import Classifier


dataset, labels = Classifier.get_json_dataset()
model = Classifier.get_model_architecture()
model.load_weights(hyperparams.model_face_classification_path)

output = model(dataset)
print(output.shape)

for i in range(output.shape[0]):
    res_index = np.argmax(output[i])
    label_index = np.argmax(labels[i])

    print(f'{output[i]} vs {labels[i]} - {"CORRECT" if res_index == label_index else "WRONG"}')
