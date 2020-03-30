import tensorflow as tf
import matplotlib.pyplot as plt


pd = tf.constant([[22,22],[22,22],[0,0]])


# Create mask from probability values
def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


# Normalize the input image and input mask
def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask

# Load image for training dataset
@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'],(122,122))

    # Add symmetric padding in order to make input image bigger. Since its size shrinks through the network,
    # we add padding so that the final output of the network would be as close as possible to the provided masks
    input_image = tf.pad(input_image, pd, 'SYMMETRIC')

    # randomly flip images
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    # Instead of 3 segments, we only consider 2 segments:
    #     - The animal in the picture
    #     - the background
    input_mask = tf.where(input_mask > 1.0, 1.0, input_mask)
    return input_image, input_mask


# Load test image
def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (122,122))

    # Add symmetric padding in order to make input image bigger. Since its size shrinks through the network,
    # we add padding so that the final output of the network would be as close as possible to the provided masks
    input_image = tf.pad(input_image, pd, 'SYMMETRIC')

    input_image, input_mask = normalize(input_image, input_mask)

    # Instead of 3 segments, we only consider 2 segments:
    #     - The animal in the picture
    #     - the background
    input_mask = tf.where(input_mask > 1.0, 1.0, input_mask)
    return input_image, input_mask


def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()
