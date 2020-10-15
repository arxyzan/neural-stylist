import tensorflow as tf
from tensorflow import keras


class InstanceNorm(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(InstanceNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        b, h, w, c = input_shape

        self.shift = self.add_weight(name='shift',
                                     shape=(c,),
                                     initializer='zeros',
                                     trainable=True)
        self.scale = self.add_weight(name='scale',
                                     shape=(c,),
                                     initializer='ones',
                                     trainable=True)
        self.epsilon = 1e-3

        self.built = True

    def call(self, inputs):
        mu, sigma_sq = tf.nn.moments(
            x=inputs, axes=[1, 2], keepdims=True)  # (T, 1, 1, C)
        normalized = (inputs - mu) / ((sigma_sq + self.epsilon) ** (0.5))
        return self.scale * normalized + self.shift


class Conv2D(keras.layers.Layer):
    def __init__(self, filter_num, filter_size, stride, padding='valid', relu=True, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.filter_num = filter_num
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.relu_activation = relu

    def build(self, input_shape):
        self.conv = keras.layers.Conv2D(
            self.filter_num, self.filter_size, self.stride, self.padding)
        self.norm = InstanceNorm()
        #self.norm = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()

        self.built = True

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.norm(x)
        if self.relu_activation:
            x = self.relu(x)

        return x


class Conv2DTranspose(keras.layers.Layer):
    def __init__(self, filter_num, filter_size, stride, padding='valid', relu=True, **kwargs):
        super(Conv2DTranspose, self).__init__(**kwargs)
        self.filter_num = filter_num
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.relu_activation = relu

    def build(self, input_shape):
        self.conv_t = keras.layers.Conv2DTranspose(
            self.filter_num, self.filter_size, self.stride, self.padding)
        self.norm = InstanceNorm()
        #self.norm = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()

        self.built = True

    def call(self, inputs):
        x = self.conv_t(inputs)
        x = self.norm(x)
        if self.relu_activation:
            x = self.relu(x)

        return x


class Res(keras.layers.Layer):
    def __init__(self, filter_num, filter_size, stride, **kwargs):
        super(Res, self).__init__(**kwargs)
        self.filter_num = filter_num
        self.filter_size = filter_size
        self.stride = stride

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filter_num, self.filter_size, self.stride)
        self.conv2 = Conv2D(self.filter_num, self.filter_size,
                            self.stride, relu=False)

        self.built = True

    def call(self, inputs):
        _, h1, w1, _ = inputs.shape

        x = self.conv1(inputs)
        x = self.conv2(x)

        _, h2, w2, _ = x.shape

        h_diff = (h1 - h2) // 2
        w_diff = (w1 - w2) // 2

        crop_inputs = inputs[:, h_diff:-h_diff, w_diff:-w_diff, :]

        return crop_inputs + x


class TransferModel(keras.models.Model):
    def __init__(self):
        super(TransferModel, self).__init__()

    def build(self, input_shape):
        self.conv1 = Conv2D(32, 9, 1, 'same')
        self.conv2 = Conv2D(64, 3, 2, 'same')
        self.conv3 = Conv2D(128, 3, 2, 'same')

        self.resid1 = Res(128, 3, 1)
        self.resid2 = Res(128, 3, 1)
        self.resid3 = Res(128, 3, 1)
        self.resid4 = Res(128, 3, 1)
        self.resid5 = Res(128, 3, 1)

        self.conv_t1 = Conv2DTranspose(64, 3, 2, 'same')
        self.conv_t2 = Conv2DTranspose(32, 3, 2, 'same')
        self.conv4 = Conv2D(3, 9, 1, 'same', relu=False)

        self.tanh1 = keras.layers.Activation('tanh')
        self.lamb1 = keras.layers.Lambda(lambda x: x * 150.0 + 255.0/2)

    def call(self, inputs):
        paddings = tf.constant([
            [0, 0],
            [40, 40],
            [40, 40],
            [0, 0]
        ])
        inputs = tf.pad(inputs, paddings, "REFLECT")

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.resid1(x)
        x = self.resid2(x)
        x = self.resid3(x)
        x = self.resid4(x)
        x = self.resid5(x)

        x = self.conv_t1(x)
        x = self.conv_t2(x)
        x = self.conv4(x)
        x = self.tanh1(x)
        x = self.lamb1(x)

        return x


def gram_matrix(input_tensor):
    result = tf.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num


def vgg_model(output_layers):
    vgg = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(layer).output for layer in output_layers]
    model = keras.models.Model([vgg.input], outputs)

    return model


class FeatureExtractor(keras.models.Model):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.style_layers = ['block1_conv1',
                             'block2_conv1',
                             'block3_conv1',
                             'block4_conv1',
                             'block5_conv1']

        self.content_layers = ['block4_conv2']

        self.num_style_layers = len(self.style_layers)

        self.vgg = vgg_model(self.style_layers + self.content_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(inputs)

        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]
        style_dict = {style_name: value for style_name,
                      value in zip(self.style_layers, style_outputs)}

        content_dict = {content_name: value for content_name,
                        value in zip(self.content_layers, content_outputs)}

        return {'style': style_dict, 'content': content_dict}


def get_style_loss(style_targets, style_features):
    style_loss = tf.math.add_n(
        [tf.math.reduce_mean((style_features[name] - style_targets[name]) ** 2)
         for name in style_features.keys()]
    )
    style_loss /= len(style_targets)  # each w_i is 0.2
    return style_loss


def get_content_loss(content_targets, content_features):
    content_loss = tf.math.add_n(
        [tf.math.reduce_mean((content_features[name] - content_targets[name]) ** 2)
         for name in content_features.keys()]
    )
    content_loss /= len(content_targets)
    return content_loss


def get_tv_loss(X):
    x_tv = X[:, :, 1:, :] - X[:, :, :-1, :]
    y_tv = X[:, 1:, :, :] - X[:, :-1, :, :]

    return tf.reduce_mean(x_tv ** 2) + tf.reduce_mean(y_tv ** 2)
