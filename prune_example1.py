import tempfile
import os

import tensorflow as tf
import numpy as np

from tensorflow import keras

# Load dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# Normalize the input image from 0-255 to 0-1
train_images, test_images = train_images / 255.0, test_images / 255.0

# expand to channel last: HWC
train_images = np.expand_dims(train_images, -1)
test_images  = np.expand_dims(test_images, -1)

# Train a model without pruning
inputs = keras.Input(shape=(28,28,1))
x = keras.layers.Conv2D(12,(3,3),activation='relu')(inputs)
x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
x = keras.layers.Flatten()(x)
outputs = keras.layers.Dense(10)(x)

model_no_pruning = keras.Model(inputs, outputs)

model_no_pruning.summary()

# Train the digit classification model
model_no_pruning.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model_no_pruning.fit(
    train_images,
    train_labels,
    epochs=4,
    validation_split=0.1,
)

loss, no_pruning_acc = model_no_pruning.evaluate(test_images, test_labels, verbose=0)
print('model_no_pruning test accuracy:', no_pruning_acc)

_, keras_file = tempfile.mkstemp('.h5')
keras.models.save_model(model_no_pruning, keras_file, include_optimizer=False)
print('Saved baseline model to:', keras_file)
model_no_pruning.save("model_no_pruning.h5")


import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

batch_size = 4
epochs = 2
validation_split = 0.1

num_images = test_images.shape[0] * (1-validation_split)
end_step   = np.ceil(num_images/batch_size).astype(np.int32) * epochs

# Define model for pruning
pruning_params = {
    'pruning_shedule':tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.5,
        final_sparsity=0.8,
        begin_step=0,
        end_step=end_step
    )
}

model_for_pruning = prune_low_magnitude(model_no_pruning, **pruning_params)

model_for_pruning.summary()

model_for_pruning.compile(
    optimizer = "adam",
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

logdir = tempfile.mkdtemp()

callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

model_for_pruning.fit(
    train_images,
    train_labels,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = validation_split,
    callbacks = callbacks
)

loss, pruning_acc = model_for_pruning.evaluate(
    test_images,
    test_labels,
    verbose=0
)
model_for_pruning.summary()
print('model_no_pruning test accuracy:', no_pruning_acc)
print('model_for_pruning test accuracy:', pruning_acc)
print("logdir:", logdir)

_, pruning_keras_file = tempfile.mkstemp('.h5')
keras.models.save_model(model_for_pruning, pruning_keras_file, include_optimizer=False)
print('Saved baseline model to:', keras_file)
print('Saved pruning model to:', pruning_keras_file)
model_for_pruning.save("model_for_pruning.h5")

model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
_, pruned_keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
print('Saved pruned Keras model to:', pruned_keras_file)
model_for_export.save("model_for_export.h5")
model_for_export.summary()