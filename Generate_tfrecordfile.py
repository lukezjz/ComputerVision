# python 3.6.8
import os
import sys
import tarfile
from six.moves import urllib
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


def download_img():
    dataset_tgz = './flower_photos.tgz'
    if not os.path.exists(dataset_tgz):
        cifar10_url = 'http://download.tensorflow.org/example_images/flower_photos.tgz'

        def __progress(block_num, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (cifar10_url, float(block_num*block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        urllib.request.urlretrieve(cifar10_url, dataset_tgz, __progress)

    with tarfile.open(dataset_tgz, 'r:gz') as tar:
        dataset_dir = tar.getnames()[0]
        if not os.path.exists(dataset_dir):
            tar.extractall()
        return dataset_dir


def split_train_test(dataset_dir, labels, test_ratio=0.2, seed=0):
    n_label = len(labels)
    train_test_dataset = {
        'train': [None] * n_label,
        'test': [None] * n_label
    }
    le = LabelEncoder()
    le.fit(labels)
    np.random.seed(seed)
    for label, index in zip(labels, le.transform(labels)):
        sub_dataset_dir = os.path.join(dataset_dir, label)
        img_files = np.array(os.listdir(sub_dataset_dir), dtype=str)
        img_file_num = len(img_files)
        img_file_index = np.arange(img_file_num)
        test_index = np.random.choice(img_file_index, int(img_file_num * test_ratio), replace=False)
        train_index = np.delete(img_file_index, test_index)
        train_test_dataset['train'][index] = [os.path.join(sub_dataset_dir, img_file) for img_file in img_files[test_index]]
        train_test_dataset['test'][index] = [os.path.join(sub_dataset_dir, img_file) for img_file in img_files[train_index]]
    return le, train_test_dataset


def imgs_to_tfrecordfile(dataset, directory, max_num=128):
    if not os.path.exists(directory):
        os.mkdir(directory)
    num = 0
    tfrecordfile_num = 0
    tfrecordfile = os.path.join(directory, 'tfrecordfile_{:0>3d}'.format(tfrecordfile_num))
    writer = tf.python_io.TFRecordWriter(tfrecordfile)
    for label in range(len(dataset)):
        for img_path in dataset[label]:
            num += 1
            if num > max_num:
                num = 0
                tfrecordfile_num += 1
                tfrecordfile = os.path.join(directory, 'tfrecordfile_{:0>3d}'.format(tfrecordfile_num))
                writer = tf.python_io.TFRecordWriter(tfrecordfile)
            img = Image.open(img_path, 'r')
            img_raw = img.tobytes()
            tf_example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    }
                )
            )
            writer.write(tf_example.SerializeToString())
    writer.close()


def main():
    dataset_dir = download_img()

    labels = []
    for item in os.listdir(dataset_dir):
        sub_dir = os.path.join(dataset_dir, item)
        if os.path.isdir(sub_dir):
            labels.append(item)

    le, train_test_dataset = split_train_test(dataset_dir, labels)
    train_dataset = train_test_dataset['train']
    test_dataset = train_test_dataset['test']

    train_dir = './train_tfrecordfiles'
    test_dir = './test_tfrecordfiles'

    imgs_to_tfrecordfile(train_dataset, train_dir)
    imgs_to_tfrecordfile(test_dataset, test_dir)


if __name__ == '__main__':
    main()
