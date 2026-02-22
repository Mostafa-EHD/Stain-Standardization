import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage as sk
from datetime import datetime
import os
import skimage.color as sk
from keras import metrics
from keras.callbacks import TensorBoard
from tensorflow.keras.applications import VGG16
from keras.optimizers import Adam

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
colorization_decoder_input_shape = (7, 7, 512)
learning_rate=0.001

result_path = os.path.join('result_path')
if os.path.exists(result_path) == False:
    os.makedirs(result_path)
path= '../root/dataset/images/'

tf.keras.backend.clear_session()
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255.0, validation_split=0.15)

train = train_gen.flow_from_directory(path, target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
                                      batch_size=BATCH_SIZE, class_mode=None, subset="training")

val = train_gen.flow_from_directory(path, target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
                                    batch_size=BATCH_SIZE, class_mode=None, subset="validation")

tf.keras.backend.clear_session()
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255.0, validation_split=0.15)

train = train_gen.flow_from_directory(path, target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
                                      batch_size=BATCH_SIZE, class_mode=None, subset="training")

val = train_gen.flow_from_directory(path, target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
                                    batch_size=BATCH_SIZE, class_mode=None, subset="validation")

train_bs = int(train.n/BATCH_SIZE)
val_bs = int(val.n/BATCH_SIZE)
print("batch_sizes", train_bs, val_bs)

def convert_lab(image):
  lab_image = sk.rgb2lab(image)
  return lab_image

def convert_rgb(image):
  rgb_image = sk.lab2rgb(image)
  return rgb_image

def plot_image(image):
  plt.figure(figsize=(12, 8))
  plt.imshow(image, cmap="gray")
  plt.grid(False)
  plt.show()
  plt.close()

# Load VGG16 model, pretrained on ImageNet, without the top classification layers
def build_vgg16_encoder(l_input):

    # Load VGG16 model, exclude the top classification layers
    vgg = VGG16(include_top=False, input_tensor=l_input)
    vgg.trainable = False

    x = l_input # Start with the input tensor
    for i in range(1, 14): # This applies layers vgg.layers[1] through vgg.layers[13]
        x = vgg.layers[i](x)
    return x

# Transformer encoder block
def transformer_encoder(inputs, num_heads, ff_dim):
    # Transformer Encoder applied on flattened VGG features
    transformer_input = layers.Reshape((-1, inputs.shape[-1]))(inputs)  # Flatten for transformer
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(transformer_input, transformer_input)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output)
    attention_output = layers.Dropout(0.1)(attention_output)

    # Feed Forward Network
    ff_output = layers.Dense(ff_dim, activation="relu")(attention_output)
    ff_output = layers.Dense(transformer_input.shape[-1])(ff_output)  # Get correct dimension

    ff_output = layers.LayerNormalization(epsilon=1e-6)(ff_output)
    ff_output = layers.Dropout(0.1)(ff_output)

    # Reshape back to image-like dimensions
    transformer_output_reshaped = layers.Reshape(inputs.shape[1:])(ff_output)

    return transformer_output_reshaped

# CNN-based decoder
def build_cnn_decoder(transformer_output_reshaped):
    output = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(transformer_output_reshaped)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dropout(0.3)(output)

    output = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dropout(0.3)(output)

    output = tf.keras.layers.UpSampling2D((2, 2))(output)

    output = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu")(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dropout(0.3)(output)

    output = tf.keras.layers.UpSampling2D((2, 2))(output)

    output = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu")(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dropout(0.3)(output)

    output = tf.keras.layers.UpSampling2D((2, 2))(output)

    output = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu")(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dropout(0.3)(output)

    output = tf.keras.layers.Conv2D(filters=2, kernel_size=(3, 3), padding="same", activation="tanh")(output)

    return output

def build_colorization_model_2(input_shape, num_heads=4, ff_dim=512):
    # L channel input (grayscale)
    l_input = layers.Input(shape=input_shape)

    vgg_encoder = build_vgg16_encoder(l_input)

    transformer_encoder_layer = transformer_encoder(vgg_encoder, num_heads=num_heads, ff_dim=ff_dim)
    cnn_decoder = build_cnn_decoder(transformer_encoder_layer)

    # Build the final model
    model = Model(inputs=l_input, outputs=cnn_decoder)
    return model

input_shape = (224, 224, 3)  # L channel input
model = build_colorization_model_2(input_shape)
optimizer = Adam(learning_rate=learning_rate)
model.summary()

x_train = []
y_train = []
for i in range(train_bs):
    if i % 100 == 0:
        print(i)
    for image in train[i]:
        try:
            lab_image = convert_lab(image)
            x_train.append(lab_image[:, :, 0])
            y_train.append(lab_image[:, :, 1:] / 128)
        except:
            print("Unexpected error. Maybe broken image.")
print("train loaded")
x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train.shape)
print(y_train.shape)

x_val = []
y_val = []
print("loading validation")
for i in range(val_bs):
    if i % 1000 == 0:
        print(i)
    for image in val[i]:
        try:
            lab_image = convert_lab(image)
            x_val.append(lab_image[:, :, 0])
            y_val.append(lab_image[:, :, 1:] / 128)
        except:
            print("Unexpected error. Maybe broken image.")
print("validation loaded")
x_val = np.array(x_val)
y_val = np.array(y_val)
print(x_val.shape)
print(y_val.shape)

x_train = x_train.reshape(x_train.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
x_val = x_val.reshape(x_val.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)

print("saving data")
np.save("path/x_train", x_train)
np.save("path/y_train", y_train)
np.save("path/x_val", x_val)
np.save("path/y_val", y_val)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=10,
                                                 min_lr=0.000001, verbose=1)
print("learning rate reducing")
monitor_es = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=False, verbose=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join("/content/drive/MyDrive/Colab Notebooks/colorization", 'best_scores', 'model_fine_ep{epoch}_valaccuracy{accuracy:.3f}.h5'),
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only= True,
                             mode='max')
log_dir = os.path.join(".",result_path, 'Graph', 'Adam')
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
print("early stopping")

np_x, np_y = [], []
for i, image in enumerate(x_train):
    image = cv2.merge((image, image, image))
    np_x.append(image)
np_x = np.array(np_x)
print(np_x.shape)
for i, image in enumerate(x_val):
    image = cv2.merge((image, image, image))
    np_y.append(image)
np_y = np.array(np_y)
print(np_y.shape)

EPOCHS = 200
border = int(len(np_x) * 0.8)
train = np_x[:border]
val = np_x[border:]
train_y = y_train[:border]
val_y = y_train[border:]
bs=16
model.compile(
    loss="binary_crossentropy",
    optimizer=optimizer,
    metrics=['accuracy']
)
history = model.fit(train, train_y, epochs=EPOCHS, validation_data=(val, val_y), verbose=1, callbacks=[reduce_lr, monitor_es,checkpoint], batch_size=8)

model.save('path/colorization_module.h5')