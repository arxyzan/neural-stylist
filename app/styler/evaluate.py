import tensorflow as tf
from tensorflow import keras

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2500)])


import os
import cv2
import json
from argparse import ArgumentParser
from model import TransferModel
from utils import read_image, write_image, optimal_size

def get_parser():
    parser = ArgumentParser()

    parser.add_argument('--type', type=str,
                        dest='type', help='content media type (image or video)', choices=['image', 'video'],
                        metavar='TYPE', required=True)
                        
    parser.add_argument('--config-file', type=str,
                    dest='config_file', help='style config',
                    metavar='CONFIG_PATH', required=True)
    
    parser.add_argument('--input', type=str,
                        dest='input', help='content path to stylish',
                        metavar='CONTENT_PATH', required=True)

    parser.add_argument('--output', type=str,
                        dest='output', help='output path to save styled result (jpg for image, avi for video)',
                        metavar='OUTPUT_PATH', required=True)

    return parser


def check_opts(opts):
    assert os.path.exists(opts.config_file), "config not found!"
    assert os.path.exists(opts.input), "input image not found!"


if __name__ == '__main__':
    parser = get_parser()
    options = parser.parse_args()
    check_opts(options)

    with open(options.config_file) as f:
        config = json.load(f)
    size = optimal_size(config['styleImagePath'], options.input)

    model = TransferModel()
    # init model weight
    ones = tf.ones((1, 256, 256, 3))
    model(ones)

    model.load_weights("{}/weights.h5".format(config['modelPath']))

    if options.type == 'image':
        content = read_image(options.input, as_4d_tensor=True, size=size)
        styled_output = model(content)
        write_image(options.output, styled_output[0] / 255.0)

    elif options.type == 'video':
        capture = cv2.VideoCapture(options.input)
        fps = capture.get(cv2.CAP_PROP_FPS)
        size = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        video_writer = cv2.VideoWriter(
            options.output,
            cv2.VideoWriter_fourcc("P", "I", "M", "1"),
            fps,
            size
        )
        
        ret, frame = capture.read()
        while ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = tf.image.convert_image_dtype(frame, tf.float32)
            frame = frame[tf.newaxis, ...]

            output = model(frame)

            out_frame = output[0] / 255.0
            out_frame = tf.clip_by_value(out_frame, clip_value_min=0.0, clip_value_max=1.0)
            out_frame = tf.image.convert_image_dtype(out_frame, tf.uint8)
            out_frame = cv2.cvtColor(out_frame.numpy(), cv2.COLOR_RGB2BGR)

            video_writer.write(out_frame)
            ret, frame = capture.read()
            
        capture.release()
        video_writer.release()
        
