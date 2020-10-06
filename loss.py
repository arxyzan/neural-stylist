import tensorflow as tf
from tensorflow import keras

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
        
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        
        return { 'style': style_dict, 'content': content_dict }


def get_style_loss(style_targets, style_features):
    style_loss = tf.math.add_n(
        [tf.math.reduce_mean((style_features[name] - style_targets[name]) ** 2) for name in style_features.keys()]
    )
    style_loss /= len(style_targets) # each w_i is 0.2
    return style_loss
    
def get_content_loss(content_targets, content_features):
    content_loss = tf.math.add_n(
        [tf.math.reduce_mean((content_features[name] - content_targets[name]) ** 2) for name in content_features.keys()]
    )
    content_loss /= len(content_targets)
    return content_loss

def get_tv_loss(X):
    x_tv = X[:, :, 1:, :] - X[:, :, :-1, :]
    y_tv = X[:, 1:, :, :] - X[:, :-1, :, :]

    return tf.reduce_mean(x_tv ** 2) + tf.reduce_mean(y_tv ** 2)


