import tensorflow as tf
import matplotlib.pyplot as plt


import hyperparams
from classifier import Classifier


dataset, labels = Classifier.get_json_dataset()
model = Classifier.get_model_architecture()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparams.classifier_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

model.summary()


cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=hyperparams.model_face_classification_path,
                                                 save_weights_only=True,
                                                 verbose=1)

history = model.fit(
    x=dataset,
    y=labels,
    batch_size=hyperparams.classifier_batch_size,
    validation_split=hyperparams.classifier_val_split,
    epochs=hyperparams.classifier_epochs,
    callbacks=[cp_callback]
)

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(hyperparams.classifier_epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
