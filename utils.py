import tensorflow as tf

'''
Function
    preprocess(image, model="mobilenet")
Args:
    image: tensor, dtype=tf.uint8
    model: string, specify the model type
Returns:
    preprocessed image, dtype=tf.float32
    
Function
    reverse_preprocess(image, model="mobilenet")
Args:
    image: tensor, dtype=tf.float32
    model: string, specify the model type
Returns:
    image with dtype=tf.uint8

Support models:
    'inception', 'inception1', 'inception2', 'inception3', 'inception4', 'inceptionresnet2_tfslim',
    'resnet', 'resnet50', 'resnet101', 'resnet152', 'resnetv2', 'resnet50v2', 'resnet101v2', 'resnet152v2', 'resnet200v2',
    'resnext', 'resnext50', 'resnext101', 'resnext50c32', 'resnext101c32', 'resnext101c64', 'wideresnet50',
    'nasnetAlarge', 'nasnetAmobile', 'pnasnetlarge',
    'vgg16', 'vgg19',
    'densenet', 'densenet121', 'densenet169', 'densenet201',
    'mobilenet', 'mobilenet25', 'mobilenet50', 'mobilenet75', 'mobilenet100',
    'mobilenetv2', 'mobilenet35v2', 'mobilenet50v2', 'mobilenet75v2', 'mobilenet100v2', 'mobilenet130v2', 'mobilenet140v2'.
'''

def preprocess(image, model="mobilenet"):
    pre_fn = __preprocess_dict__[model]
    return pre_fn(image)

def reverse_preprocess(image, model="mobilenet"):
    rev_pre_fn = __reverse_preprocess_dict__[model]
    return rev_pre_fn(image)

def tfslim_preprocess(image):
    image = tf.cast(image, tf.float32)
    image = image /  127.5
    image = image - 1.0
    return image

def reverse_tfslim_preprocess(image):
    image = image + 1.0
    image = image / 2.0
    image *= 255.
    image = tf.clip_by_value(image, 0, 255.)
    image = tf.cast(image, tf.uint8)
    return image


def bair_preprocess(image):
    image = tf.cast(image, tf.float32)
    image = tf.reverse(image, axis=[-1])
    image -= [[[[104., 117., 123.]]]]
    return image


def reverse_bair_preprocess(image):
    image += [[[[104., 117., 123.]]]]
    image = tf.reverse(image, axis=[-1])
    image = tf.clip_by_value(image, 0, 255.)
    image = tf.cast(image, tf.uint8)
    return image


def keras_resnet_preprocess(image):
    # Copied from keras and modfied
    image = tf.cast(image, tf.float32)
    image = tf.reverse(image, axis=[-1])
    image -= [[[[103.939, 116.779, 123.68]]]]
    return image


def reverse_keras_resnet_preprocess(image):
    # Reverse process of keras_resnet_preprocess
    image += [[[[103.939, 116.779, 123.68]]]]
    image = tf.reverse(image, axis=[-1])
    image = tf.clip_by_value(image, 0, 255.)
    image = tf.cast(image, tf.uint8)
    return image


def fb_preprocess(image):
    # Refer to the following Torch ResNets
    # https://github.com/facebook/fb.resnet.torch/blob/master/pretrained/classify.lua
    image = tf.cast(image, tf.float32)
    image /= 255.
    image -= [[[[0.485, 0.456, 0.406]]]]
    image /= [[[[0.229, 0.224, 0.225]]]]
    return image


def reverse_fb_preprocess(image):
    # Reverse process of fb_preprocess
    image *= [[[[0.229, 0.224, 0.225]]]]
    image += [[[[0.485, 0.456, 0.406]]]]
    image *= 255.
    image = tf.clip_by_value(image, 0, 255.)
    image = tf.cast(image, tf.uint8)
    return image


def wrn_preprocess(image):
    # Refer to the following Torch WideResNets
    # https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/main.py
    image = tf.cast(image, tf.float32)
    image /= 255.
    image -= [[[[0.491, 0.482, 0.447]]]]
    image /= [[[[0.247, 0.244, 0.262]]]]
    return image


def reverse_wrn_preprocess(image):
    # Reverse preprocess of wrn_preprocess
    image *= [[[[0.247, 0.244, 0.262]]]]
    image += [[[[0.491, 0.482, 0.447]]]]
    image *= 255.
    image = tf.clip_by_value(image, 0, 255.)
    image = tf.cast(image, tf.uint8)
    return image


# Dictionary for pre-processing functions.
__preprocess_dict__ = {
    'inception': tfslim_preprocess,
    'inception1': bair_preprocess,
    'inception2': tfslim_preprocess,
    'inception3': tfslim_preprocess,
    'inception4': tfslim_preprocess,
    'inceptionresnet2_tfslim': tfslim_preprocess,
    'resnet': keras_resnet_preprocess,
    'resnet50': keras_resnet_preprocess,
    'resnet101': keras_resnet_preprocess,
    'resnet152': keras_resnet_preprocess,
    'resnetv2': tfslim_preprocess,
    'resnet50v2': tfslim_preprocess,
    'resnet101v2': tfslim_preprocess,
    'resnet152v2': tfslim_preprocess,
    'resnet200v2': fb_preprocess,
    'resnext': fb_preprocess,
    'resnext50': fb_preprocess,
    'resnext101': fb_preprocess,
    'resnext50c32': fb_preprocess,
    'resnext101c32': fb_preprocess,
    'resnext101c64': fb_preprocess,
    'wideresnet50': wrn_preprocess,
    'nasnetAlarge': tfslim_preprocess,
    'nasnetAmobile': tfslim_preprocess,
    'pnasnetlarge': tfslim_preprocess,
    'vgg16': keras_resnet_preprocess,
    'vgg19': keras_resnet_preprocess,
    'densenet': fb_preprocess,
    'densenet121': fb_preprocess,
    'densenet169': fb_preprocess,
    'densenet201': fb_preprocess,
    'mobilenet': tfslim_preprocess,
    'mobilenet25': tfslim_preprocess,
    'mobilenet50': tfslim_preprocess,
    'mobilenet75': tfslim_preprocess,
    'mobilenet100': tfslim_preprocess,
    'mobilenetv2': tfslim_preprocess,
    'mobilenet35v2': tfslim_preprocess,
    'mobilenet50v2': tfslim_preprocess,
    'mobilenet75v2': tfslim_preprocess,
    'mobilenet100v2': tfslim_preprocess,
    'mobilenet130v2': tfslim_preprocess,
    'mobilenet140v2': tfslim_preprocess,
}

__reverse_preprocess_dict__ = {
    'inception': reverse_tfslim_preprocess,
    'inception1': reverse_bair_preprocess,
    'inception2': reverse_tfslim_preprocess,
    'inception3': reverse_tfslim_preprocess,
    'inception4': reverse_tfslim_preprocess,
    'inceptionresnet2_tfslim': reverse_tfslim_preprocess,
    'resnet': reverse_keras_resnet_preprocess,
    'resnet50': reverse_keras_resnet_preprocess,
    'resnet101': reverse_keras_resnet_preprocess,
    'resnet152': reverse_keras_resnet_preprocess,
    'resnetv2': reverse_tfslim_preprocess,
    'resnet50v2': reverse_tfslim_preprocess,
    'resnet101v2': reverse_tfslim_preprocess,
    'resnet152v2': reverse_tfslim_preprocess,
    'resnet200v2': reverse_fb_preprocess,
    'resnext': reverse_fb_preprocess,
    'resnext50': reverse_fb_preprocess,
    'resnext101': reverse_fb_preprocess,
    'resnext50c32': reverse_fb_preprocess,
    'resnext101c32': reverse_fb_preprocess,
    'resnext101c64': reverse_fb_preprocess,
    'wideresnet50': reverse_wrn_preprocess,
    'nasnetAlarge': reverse_tfslim_preprocess,
    'nasnetAmobile': reverse_tfslim_preprocess,
    'pnasnetlarge': reverse_tfslim_preprocess,
    'vgg16': reverse_keras_resnet_preprocess,
    'vgg19': reverse_keras_resnet_preprocess,
    'densenet': reverse_fb_preprocess,
    'densenet121': reverse_fb_preprocess,
    'densenet169': reverse_fb_preprocess,
    'densenet201': reverse_fb_preprocess,
    'mobilenet': reverse_tfslim_preprocess,
    'mobilenet25': reverse_tfslim_preprocess,
    'mobilenet50': reverse_tfslim_preprocess,
    'mobilenet75': reverse_tfslim_preprocess,
    'mobilenet100': reverse_tfslim_preprocess,
    'mobilenetv2': reverse_tfslim_preprocess,
    'mobilenet35v2': reverse_tfslim_preprocess,
    'mobilenet50v2': reverse_tfslim_preprocess,
    'mobilenet75v2': reverse_tfslim_preprocess,
    'mobilenet100v2': reverse_tfslim_preprocess,
    'mobilenet130v2': reverse_tfslim_preprocess,
    'mobilenet140v2': reverse_tfslim_preprocess,
}