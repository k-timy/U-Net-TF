import helper_functions as HF
from model import MyUNet
import tensorflow as tf
from tensorflow import keras as K
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import matplotlib.pyplot as plt

# Define these two variables as globals (since callbacks do not accept passed arguments at the moment)
sample_image, sample_mask = None, None


def show_predictions(my_model, dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = my_model.predict(image)
            HF.display([image[0], mask[0], HF.create_mask(pred_mask)])
    else:
        global sample_image,sample_mask
        img = sample_image[tf.newaxis, ...].numpy()
        HF.display([sample_image, sample_mask, HF.create_mask(my_model.predict(img))])


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        show_predictions(self.model)
        print('\nSample Prediction after epoch {}\n'.format(epoch + 1))


def main():

    print('Loading the dataset of pets...')
    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

    TRAIN_LENGTH = info.splits['train'].num_examples
    BATCH_SIZE = 32
    BUFFER_SIZE = 1000
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

    EPOCHS = 10
    VAL_SUBSPLITS = 5
    VALIDATION_STEPS = info.splits['test'].num_examples // BATCH_SIZE // VAL_SUBSPLITS

    train = dataset['train'].map(HF.load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test = dataset['test'].map(HF.load_image_test)

    train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_dataset = test.batch(BATCH_SIZE)
    print('The dataset is loaded successfully.')

    print('Show a sample of Image and Mask. To see how they look like')
    for image, mask in train.take(1):
        global sample_image,sample_mask
        sample_image, sample_mask = image, mask
        HF.display([sample_image, sample_mask])

    print('Create and Compile My UNet...')
    # Although each image of the dataset is 128 x 128, we consider the input size
    # as (128 + 44) x (128 + 44). The reason for doing so is that we add padding to the input images,
    # so that the segment mask would be as close as possible to 128 x 128
    my_model = MyUNet((128 + 44, 128 + 44, 3))
    my_model.compile(optimizer='adam',
                     loss=K.losses.SparseCategoricalCrossentropy(from_logits=False),
                     metrics=['accuracy'])
    print('Model is created successfully.')

    print('Show the output of the model without any training:')
    show_predictions(my_model)

    print('Training the model and track its training history:')
    model_history = my_model.fit(train_dataset, epochs=EPOCHS,
                                 steps_per_epoch=STEPS_PER_EPOCH,
                                 validation_steps=VALIDATION_STEPS,
                                 validation_data=test_dataset,
                                 callbacks=[DisplayCallback()])
    print('Training is complete.')
    # Obtaining the training and evaluation losses:
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    print('Training vs Evaluation losses:')

    # Viusalize the trend in training and evaluation losses across epochs
    epochs = range(EPOCHS)
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()

    print('Show the output of the model given 5 number of samples after it is trained:')
    show_predictions(test_dataset, 5)


if __name__ == '__main__':
    main()
