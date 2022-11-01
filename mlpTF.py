import numpy as np
import tensorflow as tf
import keras.api._v2.keras as keras
from keras import layers, models, Sequential
from keras.callbacks import EarlyStopping, CSVLogger
from keras.layers import Flatten, Dense
from matplotlib import pyplot as plt

dataset_path = "Training"  # Dataset Path
batch_size = 32
img_height = 180
img_width = 180

# ------- Building the datasets -------
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "Testing",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# ------- Normalization of the images -------
normalization_layer = layers.Rescaling(1. / 255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
class_names = train_ds.class_names
num_classes = len(class_names)

# ---- Model building ---
model_mlp = Sequential()
model_mlp.add(Flatten(input_shape=(180, 180, 3)))
model_mlp.add(Dense(350, activation='relu'))
model_mlp.add(
    Dense(num_classes, activation='softmax'))  # The last layer has to have as many neurons as the quantity of classes

# ----- Compile the model ----
model_mlp.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

model_mlp.summary()

csv_logger = CSVLogger('model2History.csv')  # Store the model training in a .csv
epochs = 12
# Train the model
history = model_mlp.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[csv_logger]
)

# Save the model
model_mlp.save('model2.h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# ------- Plot the model progression --------
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

# ----- Testing the model -----
mlp_train_loss, mlp_train_acc = model_mlp.evaluate(train_ds, verbose=0)
print('\nTrain accuracy:', np.round(mlp_train_acc, 3))
mlp_val_loss, mlp_val_acc = model_mlp.evaluate(val_ds, verbose=0)
print('\nValidation accuracy:', np.round(mlp_val_acc, 3))
mlp_test_loss, mlp_test_acc = model_mlp.evaluate(test_ds, verbose=0)
print('\nTest accuracy:', np.round(mlp_test_acc, 3))
