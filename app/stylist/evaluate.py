from utils import read_image, write_image, optimal_size
from model import TransferModel
from argparse import ArgumentParser
import json
import cv2
import os
import tensorflow as tf
from tensorflow import keras

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2500)])


def get_parser():
    parser = ArgumentParser()

    parser.add_argument('--style', type=str,
                        dest='style', help='style name',
                        metavar='STYLE', required=True)

    parser.add_argument('--input', type=str,
                        dest='input', help='input image path to style',
                        metavar='INPUT_PATH', required=True)

    return parser


def check_opts(opts):
    assert os.path.exists("models/{}".format(opts.style)), "config not found!"
    assert os.path.exists(opts.input), "input image not found!"


if __name__ == '__main__':
    parser = get_parser()
    options = parser.parse_args()
    check_opts(options)
    

    config_path = "config/{}.json".format(options.style)
    output_path = "output/{}/{}".format(options.style, options.input[-options.input[::-1].find('/'):])
    with open(config_path) as f:
        config = json.load(f)

    size = optimal_size(config['styleImagePath'], options.input)

    model = TransferModel()
    # init model weight
    ones = tf.ones((1, 256, 256, 3))
    model(ones)

    model.load_weights("{}/weights.h5".format(config['modelPath']))
    content = read_image(options.input, as_4d_tensor=True, size=size)
    styled_output = model(content)
    write_image(output_path, styled_output[0] / 255.0)
