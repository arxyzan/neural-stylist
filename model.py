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
        mu, sigma_sq = tf.nn.moments(x=inputs, axes=[1, 2], keepdims=True) # (T, 1, 1, C)
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
        self.conv = keras.layers.Conv2D(self.filter_num, self.filter_size, self.stride, self.padding)
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
        self.conv_t = keras.layers.Conv2DTranspose(self.filter_num, self.filter_size, self.stride, self.padding)
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
        self.conv2 = Conv2D(self.filter_num, self.filter_size, self.stride, relu=False)

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



class FastTransferModel(keras.models.Model):
    def __init__(self):
        super(FastTransferModel, self).__init__()

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


