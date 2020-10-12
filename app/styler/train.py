import tensorflow as tf
from tensorflow import keras

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2500)])

import os
import progressbar
import json
from argparse import ArgumentParser
from model import TransferModel, FeatureExtractor, get_style_loss, get_content_loss, get_tv_loss
from utils import read_image, write_image, image_loader

DATA_DIR = 'data/train2014'

EPOCH = 2
BATCH_SIZE = 4
LEARNING_RATE = 1e-3

STYLE_WEIGHT = 1e0
CONTENT_WEIGHT = 1e4
TV_WEIGHT = 1e6


def get_parser():
    parser = ArgumentParser()
    
    # parser.add_argument('--style-image', type=str,
    #                     dest='style_image', help='style image path',
    #                     metavar='STYLE_PATH', required=True)
    parser.add_argument('--config-file', type=str,
                        dest='config_file', help='style config',
                        metavar='CONFIG_PATH', required=True)

    parser.add_argument('--test-image', type=str,
                        dest='test_image', help='test image path, to check the model transfering effect',
                        metavar='TEST_PATH', required=True)

    parser.add_argument('--output', type=str,
                        dest='output', help='dir to save model and related output',
                        metavar='OUTPUT_DIR', required=True)

    parser.add_argument('--data', type=str,
                        dest='data', help='training images dir (default %(default)s)',
                        metavar='DATA_DIR', default=DATA_DIR)

    parser.add_argument('--epoch', type=int,
                        dest='epoch', help='num epochs (default %(default)s)',
                        metavar='EPOCH', default=EPOCH)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size (default %(default)s)',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate', help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)

    parser.add_argument('--style-weight', type=float,
                        dest='style_weight', help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)

    parser.add_argument('--content-weight', type=float,
                        dest='content_weight', help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    
    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight', help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)
    
    return parser


def check_opts(opts):
    assert os.path.exists(opts.config_file), "config not found!"
    assert os.path.exists(opts.test_image), "test image not found!"
    assert os.path.exists(opts.data), "training data not found!"
    
    assert opts.epoch > 0
    assert opts.batch_size > 0
    assert opts.learning_rate >= 0
    assert opts.content_weight >= 0
    assert opts.style_weight >= 0
    assert opts.tv_weight >= 0


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

    os.makedirs(options.output)

    style_image = read_image(config['styleImagePath'], as_4d_tensor=True)
    test_img = read_image(options.test_image, as_4d_tensor=True)

    epoch = options.epoch
    batch_size = options.batch_size
    learning_rate = options.learning_rate

    style_weight = options.style_weight
    content_weight = options.content_weight
    tv_weight = options.tv_weight
    
    dataset = tf.data.Dataset.from_generator(image_loader, tf.float32, args=[options.data, (256, 256, 3)])
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

    summary_writer = tf.summary.create_file_writer(os.path.join(options.output, 'logs'))

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
            write_image(os.path.join(options.output, '{}.jpg'.format(step_counter // auto_save_step)), test_output[0] / 255.0)

        bar.update(step_counter)

        step_counter += 1
        if step_counter > step:
            break

    bar.finish()

    transfer_model.save_weights(os.path.join(options.output, 'weights.h5'))
    
