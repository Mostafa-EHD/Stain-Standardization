from tensorflow.keras.models import load_model
import numpy as np
from skimage import color, filters
import cv2
import skimage as sk
import tensorflow as tf
import os
from leukocyte_segmentation_module import compute_channels_and_masks_batch, _thr_mask, _cmyk_from_rgb01

def onehot_to_rgb(x):
    color_dict = {0: np.broadcast_to([0, 0, 0], (1, 1, 3)), 2:np.broadcast_to([0, 200, 0], (1, 1, 3)), 1:np.broadcast_to([200, 200, 0], (1, 1, 3))}
    rgb, arg_max = np.zeros(tuple(list(x.shape[0:3]) + [3])), np.argmax(x, axis=-1)
    for color in color_dict.keys():
        rgb = np.where((arg_max==color)[:, :, :, np.newaxis], color_dict[color], rgb)
    return rgb

def predict_segmentation(model, input_rgb_batch, n_classes):
    # input_rgb_batch is expected to be (B, H, W, 3)
    nuc_ch, leu_ch, masks = compute_channels_and_masks_batch(input_rgb_batch)

    # Assemble the inputs in the same order as defined in build_unet
    # and expected by the MultiInputWrapper
    X_inputs_for_model = [
        input_rgb_batch.astype(np.float32),        # 0) RGB
        nuc_ch,                              # 1) nuc_channels
        masks["sat_otsu"],                    # 2) sat_otsu
        masks["green_min"],                   # 3) green_min
        masks["mag_iso"],                     # 4) mag_iso
        leu_ch,                               # 5) leuko_channels
        masks["yellow_li"],                   # 6) yellow_li
        masks["v_otsu"],                      # 7) v_otsu
        masks["v_yen"],                       # 8) v_yen
    ]

    # The model expects x to be a list/tuple of inputs
    model_output = model.predict(x=X_inputs_for_model, batch_size=1)
    pred = tf.keras.utils.to_categorical(np.argmax(model_output, axis=-1), num_classes=n_classes)
    rgb_prediction = onehot_to_rgb(pred)
    return rgb_prediction

colorization_module = load_model('path/colorization_module.h5')
segmentation_module = load_model('path/segmentation_module.h5')

path = "/content/dataset"
COLORIZATION_IMAGE_SIZE = (224, 224)
SEGMENTATION_IMAGE_SIZE = (128, 128)
original_images = []

for filename in os.listdir(path):
  cvimage = cv2.imread(os.path.join(path, filename))
  cvimage = cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB)
  cvimage = cv2.resize(cvimage, COLORIZATION_IMAGE_SIZE, interpolation = cv2.INTER_LINEAR)
  original_images.append(cvimage)
original_images = np.array(original_images)

luminance_images = []
ab_images = []
for image in original_images:
  lab_image = color.rgb2lab(image)
  luminance_images.append(lab_image[:, :, 0])
  ab_images.append(lab_image[:, :, 1:])
luminance_images = np.array(luminance_images)
ab_images = np.array(ab_images)
luminance_images = luminance_images.reshape(luminance_images.shape[0], COLORIZATION_IMAGE_SIZE[0], COLORIZATION_IMAGE_SIZE[1], 1)
lab_images = []
for i, image in enumerate(luminance_images):
    image = cv2.merge((image, image, image))
    lab_images.append(image)
lab_images = np.array(lab_images)

colorization_predictions = colorization_module.predict(lab_images)
colorized_images = []
for  i, image in enumerate(colorization_predictions):
    l_channel = luminance_images[i,:,:,0].astype(np.float32)
    a_channel = colorization_predictions[i,:,:,0].astype(np.float32)
    b_channel = colorization_predictions[i,:,:,1].astype(np.float32)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    rgb = color.lab2rgb(lab)
    colorized_images.append(cv2.resize(rgb, SEGMENTATION_IMAGE_SIZE, interpolation = cv2.INTER_LINEAR))
colorized_images = np.array(colorized_images)
segmented_images = predict(segmentation_module, colorized_images, n_classes=3)
