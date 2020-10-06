import tensorflow as tf
from tensorflow import keras

tf.keras.backend.clear_session()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

import os
import cv2
from argparse import ArgumentParser
from model import TransferModel
from utils import read_image, write_image

def get_parser():
    parser = ArgumentParser()
    
    parser.add_argument('--content', type=str,
                        dest='content', help='content path to stylish',
                        metavar='CONTENT_PATH', required=True)

    parser.add_argument('--output', type=str,
                        dest='output', help='output path to save styled result (jpg for image, avi for video)',
                        metavar='OUTPUT_PATH', required=True)

    parser.add_argument('--type', type=str,
                        dest='type', help='content media type (image or video)', choices=['image', 'video'],
                        metavar='TYPE', required=True)

    parser.add_argument('--weight', type=str,
                        dest='weight', help='model weight path',
                        metavar='WEIGHT', required=True)

    return parser


def check_opts(opts):
    assert os.path.exists(opts.content), "content not found!"
    assert os.path.exists(opts.weight), "weight not found!"


if __name__ == '__main__':
    parser = get_parser()
    options = parser.parse_args()
    check_opts(options)

    model = TransferModel()
    # init model weight
    ones = tf.ones((1, 256, 256, 3))
    model(ones)

    model.load_weights(options.weight)

    if options.type == 'image':
        content = read_image(options.content, as_4d_tensor=True)
        styled_output = model(content)
        write_image(options.output, styled_output[0] / 255.0)

    elif options.type == 'video':
        capture = cv2.VideoCapture(options.content)
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
        
