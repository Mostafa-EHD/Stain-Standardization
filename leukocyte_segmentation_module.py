import numpy as np
from skimage import color, filters
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Conv2D, MaxPooling2D, Add, SpatialDropout2D, concatenate, BatchNormalization, UpSampling2D, Activation, ReLU, Lambda, Concatenate

import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam
from keras import metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from datetime import datetime
import numpy as np
import os
from matplotlib import pyplot as plt
from skimage import color, filters

"""# Initialisation of global variables and hyperparameters used throughout the notebook"""

stop_val = 50
width = 128
height = 128
epochs = 500
bs=32
n_classes = 3
model_id= datetime.now().strftime("%Y%m%d_%H%M%S")
seed = 2

"""# Preprocessing Pipeline: Channel Extraction and Threshold-Based Mask Computation"""

def _thr_mask(img1, thr_func):
    # img1: (H,W,1) float32
    img2 = img1[..., 0]
    try:
        t = thr_func(img2)
    except RuntimeError as e:
        if "Unable to find two maxima in histogram" in str(e):
            # Fallback: if threshold_minimum fails, return a mask of all zeros.
            # This is a safe fallback for image data where a clear bimodal histogram isn't present.
            return np.zeros_like(img1, dtype=np.float32)
        else:
            raise # Re-raise other RuntimeErrors
    return (img2 > t).astype(np.float32)[..., None]  # (H,W,1)

def _cmyk_from_rgb01(rgb01):
    # rgb01: (H,W,3) float32 in [0,1]
    K = 1.0 - np.max(rgb01, axis=-1, keepdims=True)
    denom = (1.0 - K + 1e-7)
    C = (1.0 - rgb01[..., 0:1] - K) / denom
    M = (1.0 - rgb01[..., 1:2] - K) / denom
    Y = (1.0 - rgb01[..., 2:3] - K) / denom
    return np.concatenate([C, M, Y, K], axis=-1).astype(np.float32)

def compute_channels_and_masks_batch(rgb_uint8_batch):
    B, H, W, _ = rgb_uint8_batch.shape
    rgb01_batch = rgb_uint8_batch.astype(np.float32)
    if rgb01_batch.max() > 1.5:
        rgb01_batch /= 255.0

    nuc_channels   = np.zeros((B, H, W, 6), np.float32)
    leuko_channels = np.zeros((B, H, W, 6), np.float32)

    sat_otsu  = np.zeros((B, H, W, 1), np.float32)
    green_min = np.zeros((B, H, W, 1), np.float32)
    mag_iso   = np.zeros((B, H, W, 1), np.float32)

    yellow_li = np.zeros((B, H, W, 1), np.float32)
    v_otsu    = np.zeros((B, H, W, 1), np.float32)
    v_yen     = np.zeros((B, H, W, 1), np.float32)

    for i in range(B):
        rgb01 = rgb01_batch[i]

        hsv = color.rgb2hsv(rgb01).astype(np.float32)   # (H,W,3)
        lab = color.rgb2lab(rgb01).astype(np.float32)   # (H,W,3)
        luv = color.rgb2luv(rgb01).astype(np.float32)   # (H,W,3)
        cmyk = _cmyk_from_rgb01(rgb01)                  # (H,W,4)

        # Common channels
        hue = hsv[..., 0:1]
        sat = hsv[..., 1:2]
        green = rgb01[..., 1:2]

        # Nucleus channels:
        magenta = cmyk[..., 1:2]
        L_lab   = lab[..., 0:1]
        a_lab   = lab[..., 1:2]
        L_luv   = luv[..., 0:1]
        nuc_channels[i] = np.concatenate([magenta, L_lab, a_lab, hue, sat, L_luv], axis=-1)

        # Leukocyte channels:
        cyan   = cmyk[..., 0:1]
        yellow = cmyk[..., 2:3]
        b_lab  = lab[..., 2:3]
        v_luv  = luv[..., 2:3]
        leuko_channels[i] = np.concatenate([cyan, yellow, b_lab, hue, sat, v_luv], axis=-1)

        # Masks (CPU thresholds)
        sat_otsu[i]  = _thr_mask(sat,      filters.threshold_otsu)
        green_min[i] = _thr_mask(green,    filters.threshold_minimum)
        mag_iso[i]   = _thr_mask(magenta,  filters.threshold_isodata)

        yellow_li[i] = _thr_mask(yellow,   filters.threshold_li)
        v_otsu[i]    = _thr_mask(v_luv,    filters.threshold_otsu)
        v_yen[i]     = _thr_mask(v_luv,    filters.threshold_yen)

    masks = {
        "sat_otsu": sat_otsu,
        "green_min": green_min,
        "mag_iso": mag_iso,
        "yellow_li": yellow_li,
        "v_otsu": v_otsu,
        "v_yen": v_yen,
    }
    return nuc_channels, leuko_channels, masks

"""# MultiInputWrapper: Custom Data Generator for Multi-Branch U-Net"""

class MultiInputWrapper:
    def __init__(self, base_iter):
        self.base_iter = base_iter

    def __iter__(self):
        return self

    def __next__(self):
        rgb_batch, y_batch = next(self.base_iter)
        nuc_ch, leu_ch, masks = compute_channels_and_masks_batch(rgb_batch)

        X = [
            rgb_batch.astype(np.float32),        # 0)
            nuc_ch,                              # 1)
            masks["sat_otsu"],                    # 2)
            masks["green_min"],                   # 3)
            masks["mag_iso"],                     # 4)
            leu_ch,                               # 5)
            masks["yellow_li"],                   # 6)
            masks["v_otsu"],                      # 7)
            masks["v_yen"],                       # 8)
        ]
        return X, y_batch

    def __len__(self):
        # important pour steps_per_epoch si Keras le demande
        return len(self.base_iter)

def pooling(layer):
    return MaxPooling2D(pool_size=(2, 2))(layer)

def dropout(layer):
    return SpatialDropout2D(0.2)(layer)

def build_block(input_layer, filters, norm=True, k=(3, 3)):
    layer = Conv2D(filters, kernel_size=k, padding='same', use_bias=not norm, kernel_initializer='glorot_normal')(input_layer)
    if norm:
        layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    return layer

def residual_block(input_tensor, num_filters, kernel_size=(3, 3), stride=(1, 1)):
    # First convolution
    x = Conv2D(num_filters, kernel_size=kernel_size, strides=stride, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second convolution
    x = Conv2D(num_filters, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    # Shortcut connection (identity mapping)
    if stride != (1, 1) or input_tensor.shape[-1] != num_filters:
        shortcut = Conv2D(num_filters, kernel_size=(1, 1), strides=stride, padding='same')(input_tensor)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = input_tensor

    # Add the shortcut to the main path
    x = Add()([x, shortcut])
    x = ReLU()(x)

    return x

"""# RGB Encoder Block for Multi-Input U-Net Architecture"""

def rgb_encoder_bloc(image_input, n_filters=64):

    conv_1 = build_block(image_input, n_filters) #(32, 32, 64)
    conv_2 = build_block(conv_1, n_filters) #(32, 32, 64)
    conv_3 = build_block(conv_2, n_filters) #(32, 32, 64)
    pool_1 = pooling(conv_3)#(16, 16, 64)
    drop_1 = dropout(pool_1)

    conv_4 = build_block(drop_1, n_filters * 2)#(16, 16, 64)
    conv_5 = build_block(conv_4, n_filters * 2)#(16, 16, 64)
    conv_6 = build_block(conv_5, n_filters * 2)#(16, 16, 64)
    pool_2 = pooling(conv_6)#(8, 8, 64)
    drop_2 = dropout(pool_2)

    conv_7 = build_block(drop_2, n_filters * 4)#(8, 8, 256)
    conv_8 = build_block(conv_7, n_filters * 4) #(8, 8, 256)
    conv_9 = build_block(conv_8, n_filters * 4) #(8, 8, 256)
    pool_3 = pooling(conv_9)#(4, 4, 256)
    drop_3 = dropout(pool_3)

    conv_10 = build_block(drop_3, n_filters * 8)#(4, 4, 512)
    conv_11 = build_block(conv_10, n_filters * 8) #(4, 4, 512)
    conv_12 = build_block(conv_11, n_filters * 8) #(4, 4, 512)
    pool_4 = pooling(conv_12)#(4, 4, 512)

    return  pool_4, conv_3, conv_6, conv_9, conv_12

"""# Mask-Guided Residual Leukocyte Encoder for Multi-Input U-Net Architecture"""

def leukocyte_encoder_bloc(image_input, color_channels, yellow_li, v_otsu, v_yen, n_filters=64):
    resid1 = residual_block(input_tensor=Concatenate(axis=-1)([color_channels, yellow_li]),
                            num_filters=n_filters)
    resid1 = pooling(resid1)

    conv_1 = build_block(color_channels, n_filters)
    conv_2 = build_block(conv_1, n_filters)
    conv_3 = build_block(conv_2, n_filters)
    pool_1 = pooling(conv_3)
    drop_1 = dropout(pool_1)
    output_tensor1 = Add()([resid1, drop_1])

    v_otsu_pooled = pooling(v_otsu)

    resid2 = residual_block(input_tensor=Concatenate(axis=-1)([output_tensor1, v_otsu_pooled]), num_filters=n_filters*2)
    resid2 = pooling(resid2)

    conv_4 = build_block(output_tensor1, n_filters * 2)
    conv_5 = build_block(conv_4, n_filters * 2)
    conv_6 = build_block(conv_5, n_filters * 2)
    pool_2 = pooling(conv_6)
    drop_2 = dropout(pool_2)
    output_tensor2 = Add()([resid2, drop_2])

    v_yen_pooled = pooling(v_yen)
    v_yen_pooled = pooling(v_yen_pooled)

    resid3 = residual_block(input_tensor=Concatenate(axis=-1)([output_tensor2, v_yen_pooled]), num_filters=n_filters*4)
    resid3 = pooling(resid3)

    conv_7 = build_block(output_tensor2, n_filters * 4)
    conv_8 = build_block(conv_7, n_filters * 4)
    conv_9 = build_block(conv_8, n_filters * 4)
    pool_3 = pooling(conv_9)
    drop_3 = dropout(pool_3)
    output_tensor3 = Add()([resid3, drop_3])

    conv_10 = build_block(output_tensor3, n_filters * 8)
    conv_11 = build_block(conv_10, n_filters * 8)
    conv_12 = build_block(conv_11, n_filters * 8)
    pool_4 = pooling(conv_12)

    return pool_4, conv_3, conv_6, conv_9, conv_12

"""# Mask-Guided Residual Nucleus Encoder for Multi-Input U-Net Architecture"""

def nuc_encoder_bloc(image_input, color_channels, sat_otsu, green_min, mag_iso, n_filters=64):
    resid1 = residual_block(
        input_tensor=Concatenate(axis=-1)([color_channels, sat_otsu]),
        num_filters=n_filters
    )
    resid1 = pooling(resid1)

    conv_1 = build_block(color_channels, n_filters)
    conv_2 = build_block(conv_1, n_filters)
    conv_3 = build_block(conv_2, n_filters)
    pool_1 = pooling(conv_3)
    drop_1 = dropout(pool_1)
    output_tensor1 = Add()([resid1, drop_1])

    green_pooled = pooling(green_min)

    resid2 = residual_block(
        input_tensor=Concatenate(axis=-1)([output_tensor1, green_pooled]),
        num_filters=n_filters * 2
    )
    resid2 = pooling(resid2)

    conv_4 = build_block(output_tensor1, n_filters * 2)
    conv_5 = build_block(conv_4, n_filters * 2)
    conv_6 = build_block(conv_5, n_filters * 2)
    pool_2 = pooling(conv_6)
    drop_2 = dropout(pool_2)
    output_tensor2 = Add()([resid2, drop_2])

    mag_pooled = pooling(mag_iso)
    mag_pooled = pooling(mag_pooled)

    resid3 = residual_block(
        input_tensor=Concatenate(axis=-1)([output_tensor2, mag_pooled]),
        num_filters=n_filters * 4
    )
    resid3 = pooling(resid3)

    conv_7 = build_block(output_tensor2, n_filters * 4)
    conv_8 = build_block(conv_7, n_filters * 4)
    conv_9 = build_block(conv_8, n_filters * 4)
    pool_3 = pooling(conv_9)
    drop_3 = dropout(pool_3)
    output_tensor3 = Add()([resid3, drop_3])

    conv_10 = build_block(output_tensor3, n_filters * 8)
    conv_11 = build_block(conv_10, n_filters * 8)
    conv_12 = build_block(conv_11, n_filters * 8)
    pool_4 = pooling(conv_12)

    return pool_4, conv_3, conv_6, conv_9, conv_12

def decoder(n_classes=3, n_filters=64, conv_2=None, conv_4=None, conv_6=None, conv_8=None, conv_10=None):
    upsp_1 = UpSampling2D(size=(2, 2))(conv_10) #(-1, 8, 8, 64)
    upsp_1 = concatenate([upsp_1, conv_8]) #(-1, 8, 8, 192)
    conv_11 = build_block(upsp_1, n_filters * 8) #(-1, 8, 8, 64)
    conv_12 = build_block(conv_11, n_filters * 8)
    conv_13 = build_block(conv_12, n_filters * 8)
    drop_7 = dropout(conv_13)

    upsp_3 = UpSampling2D(size=(2, 2))(drop_7) #(-1, 16, 16, 64)
    upsp_3 = concatenate([upsp_3, conv_6]) #(-1, 16, 16, 192)
    conv_14 = build_block(upsp_3, n_filters * 4) #(-1, 16, 16, 64)
    conv_15 = build_block(conv_14, n_filters * 4)
    conv_16 = build_block(conv_15, n_filters * 4)
    drop_8 = dropout(conv_16)

    upsp_4 = UpSampling2D(size=(2, 2))(drop_8) #(-1, 32, 32, 64)
    upsp_4 = concatenate([upsp_4, conv_4])#(-1, 32, 32, 92)
    conv_17 = build_block(upsp_4, n_filters * 2)#(-1, 32, 32, 32)
    conv_18 = build_block(conv_17, n_filters * 2)
    conv_19 = build_block(conv_18, n_filters * 2)
    drop_9 = dropout(conv_19)

    upsp_5 = UpSampling2D(size=(2, 2))(drop_9) #(-1, 32, 32, 64)
    upsp_5 = concatenate([upsp_5, conv_2])#(-1, 32, 32, 92)
    conv_20 = build_block(upsp_5, n_filters)#(-1, 32, 32, 32)
    conv_21 = build_block(conv_20, n_filters)
    conv_22 = build_block(conv_21, n_filters)
    drop_10 = dropout(conv_22)

    output = Conv2D(n_classes, (1, 1), kernel_initializer='glorot_normal', activation='softmax')(drop_10)
    return output

"""# Fusion-Based Multi-Encoder U-Net Architecture for Leukocyte Segmentation"""

def build_unet(input_shape=(128, 128, 3), n_filters=64, n_classes=4):
    h, w, c = input_shape
    rgb_in = Input(input_shape, name="rgb")
    nuc_ch_in      = Input((h,w,6), name="nuc_channels") # Changed from (128,128,6)
    sat_otsu_in    = Input((h,w,1), name="sat_otsu") # Changed from (128,128,1)
    green_min_in   = Input((h,w,1), name="green_min") # Changed from (128,128,1)
    mag_iso_in     = Input((h,w,1), name="mag_iso") # Changed from (128,128,1)

    leu_ch_in      = Input((h,w,6), name="leuko_channels") # Changed from (128,128,6)
    yellow_li_in   = Input((h,w,1), name="yellow_li") # Changed from (128,128,1)
    v_otsu_in      = Input((h,w,1), name="v_otsu") # Changed from (128,128,1)
    v_yen_in       = Input((h,w,1), name="v_yen") # Changed from (128,128,1)

    module_1, conv_2_1, conv_4_1, conv_6_1, conv_8_1 = rgb_encoder_bloc(image_input=rgb_in, n_filters=n_filters)
    module_2, conv_2_2, conv_4_2, conv_6_2, conv_8_2 = nuc_encoder_bloc(image_input=rgb_in,
                                                                        color_channels=nuc_ch_in,
                                                                        sat_otsu=sat_otsu_in,
                                                                        green_min=green_min_in,
                                                                        mag_iso=mag_iso_in,
                                                                        n_filters=n_filters)
    module_3, conv_2_3, conv_4_3, conv_6_3, conv_8_3 = leukocyte_encoder_bloc(image_input=rgb_in,
                                                                        color_channels=leu_ch_in,
                                                                        yellow_li=yellow_li_in,
                                                                        v_otsu=v_otsu_in,
                                                                        v_yen=v_yen_in,
                                                                        n_filters=n_filters)
    encoder = concatenate([module_1, module_2, module_3])
    conv_9 = build_block(encoder, n_filters * 16)#(4, 4, 64)
    conv_10 = build_block(conv_9, n_filters * 16) #(4, 4, 64)
    output = decoder(conv_2=concatenate([conv_2_1, conv_2_2, conv_2_3]),
                     conv_4=concatenate([conv_4_1, conv_4_2, conv_4_3]),
                     conv_6=concatenate([conv_6_1, conv_6_2, conv_6_3]),
                     conv_8=concatenate([conv_8_1, conv_8_2, conv_8_3]),
                     conv_10=conv_10, n_classes=n_classes, n_filters=n_filters)
    model = Model(inputs=[rgb_in, nuc_ch_in, sat_otsu_in, green_min_in, mag_iso_in, leu_ch_in, yellow_li_in, v_otsu_in, v_yen_in],
                  outputs=output)

    optimizer = Adam(learning_rate=1e-4)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=['accuracy', metrics.MeanIoU(num_classes=n_classes)]
    )
    model.summary()

    return model

"""# Training Strategy and Model Saving Configuration

"""

stopper = EarlyStopping(monitor = 'val_accuracy',
                            min_delta=0,
                            patience=stop_val,
                            verbose=1,
                            mode='max')


model_filename = 'segmentation_module.h5'

input_shape = (width, height, 3)

result_path = '/results_path'
if os.path.exists(result_path) == False:
    os.makedirs(result_path)

"""# Training Pipeline: Augmentation, Checkpointing, and Monitoring"""

final_model = build_unet(input_shape=input_shape, n_classes=n_classes)
final_model.summary()

checkpoint = ModelCheckpoint(os.path.join(result_path, model_filename),
                            monitor='val_accuracy',
                            verbose=1,
                            save_best_only= True,
                            mode='max')

datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
datagen.fit(train_inputs, augment=True, seed=seed)

train_base = datagen.flow(train_inputs, train_labels, batch_size=bs, seed=seed)
val_base   = ImageDataGenerator().flow(val_inputs, val_labels, batch_size=bs, shuffle=False)

train_gen = MultiInputWrapper(train_base)
val_gen   = MultiInputWrapper(val_base)

log_dir = os.path.join(".",result_path, 'Graph', 'Adam')
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

callback_list = [checkpoint, tensorboard_callback, stopper]

"""# Dataset Preparation and Model Training Execution"""

# Define a generator function that the tf.data.Dataset can use
def create_multimodel_generator(input_wrapper_instance):
    for X_batch, y_batch in input_wrapper_instance:
        # Ensure y_batch is float32 for Keras, if it's not already
        # X_batch is already a list of numpy arrays from MultiInputWrapper
        yield tuple(X_batch), y_batch.astype(np.float32) # Convert X_batch to tuple

# Define the output signature for the tf.data.Dataset
# The shapes are (batch_size, height, width, channels), where batch_size=None indicates variable batch size.
output_data_signature = (
    ( # This expects a tuple of inputs
        tf.TensorSpec(shape=(None, height, width, 3), dtype=tf.float32), # RGB
        tf.TensorSpec(shape=(None, height, width, 6), dtype=tf.float32), # nuc_channels
        tf.TensorSpec(shape=(None, height, width, 1), dtype=tf.float32), # sat_otsu
        tf.TensorSpec(shape=(None, height, width, 1), dtype=tf.float32), # green_min
        tf.TensorSpec(shape=(None, height, width, 1), dtype=tf.float32), # mag_iso
        tf.TensorSpec(shape=(None, height, width, 6), dtype=tf.float32), # leuko_channels
        tf.TensorSpec(shape=(None, height, width, 1), dtype=tf.float32), # yellow_li
        tf.TensorSpec(shape=(None, height, width, 1), dtype=tf.float32), # v_otsu
        tf.TensorSpec(shape=(None, height, width, 1), dtype=tf.float32), # v_yen
    ),
    tf.TensorSpec(shape=(None, height, width, n_classes), dtype=tf.float32) # y_batch (labels)
)

# Create tf.data.Dataset objects from your MultiInputWrapper instances
train_dataset = tf.data.Dataset.from_generator(
    lambda: create_multimodel_generator(train_gen),
    output_signature=output_data_signature
)

val_dataset = tf.data.Dataset.from_generator(
    lambda: create_multimodel_generator(val_gen),
    output_signature=output_data_signature
)

history = final_model.fit(
    x=train_dataset,
    steps_per_epoch=int(np.ceil(train_labels.shape[0] / bs))*150,
    epochs=epochs,
    validation_data=val_dataset, # Use the tf.data.Dataset
    validation_steps=int(np.ceil(val_labels.shape[0] / bs)),
    callbacks=callback_list,
    verbose=1
)
# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.ioff()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join(result_path, 'accuracy.jpg'))
plt.close()
# summarize history for loss
plt.ioff()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join(result_path, 'loss.jpg'))
plt.close()
final_model.save(os.path.join(result_path, model_filename))
    file_name = path[i]
    img = cv2.imread(os.path.join(dataset_path, file_name[:3], file_name))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation = cv2.INTER_LINEAR)
    prediction = predict(final_model, np.expand_dims(img, axis=0), n_classes) # Explicitly add batch dimension
    y = cv2.imread(os.path.join(labels_path, file_name[:3], path2[i]))
    y = cv2.cvtColor(y,cv2.COLOR_BGR2RGB)
    y = cv2.resize(y, size, interpolation = cv2.INTER_LINEAR)
    plt.ioff()
    plt.figure(figsize=(16, 4), dpi = 100)
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.xlabel('original', fontsize=10)
    plt.yticks([])
    plt.xticks([])
    plt.subplot(1, 3, 2)
    plt.imshow(y, cmap='gray')
    plt.xlabel('Ground Truth', fontsize=10)
    plt.yticks([])
    plt.xticks([])
    plt.subplot(1, 3, 3)
    plt.imshow(prediction[0], cmap='gray') # Access the first (and only) image in the batch
    plt.xlabel('Segmentation', fontsize=10)
    plt.yticks([])
    plt.xticks([])
    plt.show()