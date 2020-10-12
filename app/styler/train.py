import tensorflow as tf
from tensorflow import keras

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2500)])

import os
import progressbar
from shutil import rmtree
import json
from argparse import ArgumentParser
from model import TransferModel, FeatureExtractor, get_style_loss, get_content_loss, get_tv_loss
from utils import read_image, write_image, image_loader, compute_size

DATA_DIR = 'data/train2014'

EPOCH = 2
BATCH_SIZE = 4
LEARNING_RATE = 1e-3

STYLE_WEIGHT = 1e0
CONTENT_WEIGHT = 1e4
TV_WEIGHT = 1e6


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--config-file', type=str,
                        dest='config_file', help='style config',
                        metavar='CONFIG_PATH', required=True)
    
    return parser


def check_opts(opts):
    assert os.path.exists(opts.config_file), "config not found!"


def get_progress_bar():
    widgets = [
        progressbar.Percentage(),
        ' ', progressbar.Bar(left='[', right=']'),
        ' ', progressbar.AnimatedMarker(),
        ' ', progressbar.ETA(),
        ', step ',
        progressbar.SimpleProgress(),
    ]
    
    return progressbar.ProgressBar(max_value=step, widgets=widgets)


if __name__ == '__main__':
    parser = get_parser()
    options = parser.parse_args()
    check_opts(options)

    with open(options.config_file) as f:
        config = json.load(f)

    if os.path.exists(config['modelPath']):
        rmtree(config['modelPath'])
    os.makedirs(config['modelPath'])

    size = compute_size(config)
    style_image = read_image(config['styleImagePath'], as_4d_tensor=True)
    test_img = read_image(config['testImagePath'], as_4d_tensor=True, size=size)

    epoch = config['epoch']
    batch_size = config['batchSize']
    learning_rate = config['learningRate']

    style_weight = config['styleWeight']
    content_weight = config['contentWeight']
    tv_weight = config['tvWeight']
    
    dataset = tf.data.Dataset.from_generator(image_loader, tf.float32, args=[config['datasetPath'], (256, 256, 3)])
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    data_size_per_epoch = 4e4
    step = data_size_per_epoch * epoch // batch_size
    auto_save_step = step // 100

    transfer_model = TransferModel()
    feature_extractor = FeatureExtractor()
    style_targets = feature_extractor(tf.constant(style_image * 255.0))['style']

    style_loss_metric = keras.metrics.Mean(name='style_loss')
    content_loss_metric = keras.metrics.Mean(name='content_loss')
    tv_loss_metric = keras.metrics.Mean(name='tv_loss')

    summary_writer = tf.summary.create_file_writer(os.path.join(config['modelPath'], 'logs'))

    bar = get_progress_bar()

    @tf.function()
    def train_step(X):
        with tf.GradientTape() as tape:
            content_targets = feature_extractor(X * 255.0)['content']

            outputs = transfer_model(X)
            
            features = feature_extractor(outputs)
            style_features = features['style']
            content_features = features['content']

            style_loss = style_weight * get_style_loss(style_targets, style_features)
            content_loss = content_weight * get_content_loss(content_targets, content_features)
            tv_loss = tv_weight * get_tv_loss(outputs)

            loss = style_loss + content_loss + tv_loss
            
        grad = tape.gradient(loss, transfer_model.trainable_variables)
        optimizer.apply_gradients(zip(grad, transfer_model.trainable_variables))

        style_loss_metric(style_loss)
        content_loss_metric(content_loss)
        tv_loss_metric(tv_loss)

    bar.start()
    step_counter = 0

    for X in dataset.repeat().batch(batch_size):
        style_loss_metric.reset_states()
        content_loss_metric.reset_states()
        tv_loss_metric.reset_states()
        
        train_step(X)

        with summary_writer.as_default():
            tf.summary.scalar('style loss', style_loss_metric.result(), step=step_counter)
            tf.summary.scalar('content loss', content_loss_metric.result(), step=step_counter)
            tf.summary.scalar('tv loss', tv_loss_metric.result(), step=step_counter)

        if step_counter % auto_save_step == 0:
            test_output = transfer_model(test_img)
            write_image(os.path.join(config['modelPath'], '{}.jpg'.format(step_counter // auto_save_step)), test_output[0] / 255.0)

        bar.update(step_counter)

        step_counter += 1
        if step_counter > step:
            break

    bar.finish()

    transfer_model.save_weights(os.path.join(config['modelPath'], 'weights.h5'))
    
