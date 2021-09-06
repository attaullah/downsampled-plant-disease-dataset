import os
import sys
import numpy as np
from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_enum(name="name", default="plant32", enum_values=["plant32", "plant64", "plant96"],
                  help="dataset name")
flags.DEFINE_string(name="data_dir", default="data/", help="path to data_dir containing train/ and test/ images of "
                                                           "full-dataset.")
flags.DEFINE_string(name="output_dir", default="data/", help=" resultant .npz file will be saved at this location.")


def count(path):
    """
    returns number of files in dir and sub-dirs.
    :param path: to directory
    :return: return count
    """
    counter = 0
    for pack in os.walk(path):
        for _ in pack[2]:
            counter += 1
    return counter


def prepare_plant_village_dataset(name, data_dir, output_dir):
    """
    Loads original PlantVillage dataset once and converts to the required shape and saves as numpy array
    :param output_dir: for saving .npz file
    :param data_dir: containing train/ and test/ directories of full plant villae dataset.
    :param name: name with resolution size
    :return: Training and test images and labels
    """
    test_dir = data_dir + '/test'
    train_dir = data_dir + '/train'
    print('Dataset not found  at=', data_dir, '. Creating npz file, this will be done once.')
    val_size = count(test_dir)
    train_size = count(train_dir)
    if '32' in name:
        pixels = 32
    elif '64' in name:
        pixels = 64
    elif '96' in name:
        pixels = 96
    else:
        print(name, ": dataset not supported.")
        sys.exit(1)

    import tensorflow as tf
    image_size = (pixels, pixels)
    aug = tf.keras.preprocessing.image.ImageDataGenerator()
    test_generator = aug.flow_from_directory(test_dir, color_mode="rgb", class_mode='sparse',
                                             target_size=image_size, batch_size=val_size)
    train_generator = aug.flow_from_directory(train_dir, color_mode="rgb", target_size=image_size,
                                              batch_size=train_size, class_mode='sparse')
    train_images, train_labels = train_generator.next()
    test_images, test_labels = test_generator.next()

    save_path = output_dir + '/' + name + '.npz'
    print('saving as: ', save_path)
    np.savez_compressed(save_path, train_images=train_images, train_labels=train_labels,
                        test_images=test_images, test_labels=test_labels)
    return train_images, train_labels, test_images, test_labels


def load_dataset(argv):
    """
    Loads PlantVillage dataset.
    :return: Training and test images and labels
    """
    name = FLAGS.name
    file_path = FLAGS.data_dir + '/' + name+'.npz'
    if os.path.exists(file_path):  # if name.npz file already exists.
        npzfile = np.load(file_path)
        train_images, train_labels = npzfile['train_images'], npzfile['train_labels']
        test_images, test_labels = npzfile['test_images'], npzfile['test_labels']
    else:
        train_images, train_labels, test_images, test_labels = prepare_plant_village_dataset(name,
                                                                                             data_dir=FLAGS.data_dir,
                                                                                             output_dir=FLAGS.output_dir)
    print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)
    # return train_images, train_labels, test_images, test_labels


if __name__ == '__main__':
    FLAGS.alsologtostderr = True
    app.run(load_dataset)
