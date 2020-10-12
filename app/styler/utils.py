import tensorflow as tf
from imutils import paths
from PIL import Image

def compute_size(config):
    style_path = config['styleImagePath']
    content_path = config['contentImagePath']
    style = Image.open(style_path)
    content = Image.open(content_path)
    style_w, style_h = style.size
    content_w, content_h = content.size
    ratio = style_w/content_w
    return (int(content_w * ratio), int(content_h * ratio))


def read_image(image_path, as_4d_tensor=False, size=None):
    i = tf.io.read_file(image_path)
    i = tf.image.decode_image(i, channels=3) # HWC
    i = tf.image.convert_image_dtype(i, tf.float32) # uint8 [0, 255] -> float32 [0, 1]
    if as_4d_tensor:
        i = i[tf.newaxis, ...]
    if size is not None:
        i = tf.image.resize(i, size) 
    return i

def write_image(image_path, image, size=None):
    if size is not None:
        image = tf.image.resize(image, size)
    i = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    i = tf.image.convert_image_dtype(i, tf.uint8) # float32 [0, 1] -> uint8 [0, 255]
    i = tf.image.encode_jpeg(i)
    tf.io.write_file(image_path, i)

def image_loader(img_dir, output_shape):
    for image_path in paths.list_images(img_dir.decode('utf-8')):
        i = read_image(image_path)
        yield tf.image.resize(i, output_shape[:-1])

