import numpy as np
from scipy.misc import imresize


def _colorize_images(images):
    def _handle_args():
        if type(images) is np.ndarray:
            if len(images.shape) > 2:
                return _colorize_images(list(images))

    err_arg = _handle_args()
    if err_arg is not None:
        return err_arg

    color_images = []
    for i, image in enumerate(images):
        if len(image.shape) == 2:
            color_images.append(np.repeat(np.expand_dims(image, -1), 3, -1))
        elif len(image.shape) == 3:
            if image.shape[-1] == 3:
                color_images.append(image)
            elif image.shape[-1] == 1:
                color_images.append(np.repeat(image, 3, -1))
            else:
                raise ValueError('Incorrect image dimensions for image at {}'.format(i))
        else:
            raise ValueError('Incorrect image dimensions for image at {}'.format(i))

    return color_images


def _resize_images(images, shape='smean', interp='bilinear', mode=None):
    def _resolve_shape():
        nonlocal shape
        shapes = np.array([image.shape[:-1] for image in images])

        # Make all the shapes square
        if shape[0] == 's':
            shapes = np.array([[int(np.sqrt(np.prod(s)))] * 2 for s in shapes])
            shape = shape[1:]

        if shape == 'min':
            shape = shapes.min(0)
        elif shape == 'max':
            shape = shapes.max(0)
        elif shape == 'mean':
            shape = shapes.mean(0)

        shape = shape.astype(np.uint)

    def _handle_args():
        nonlocal shape
        if any(len(image.shape) != 3 for image in images):
            raise ValueError('All images must have 3 dimensions.')
        if type(shape) not in (tuple, list):
            if type(shape) is not str:
                raise TypeError('shape must be either a tuple, list or string.')
            if shape not in ['min', 'max', 'mean', 'smin', 'smax', 'smean']:
                raise ValueError("shape must be one of ('min', 'max', 'mean', 'smin', 'smax', 'smean')")

            _resolve_shape()
        elif any(type(s) is not int or s <= 0 for s in shape):
            raise ValueError('shape must have positive integer elements')

    err_arg = _handle_args()
    if err_arg is not None:
        return err_arg

    return [imresize(image, shape, interp, mode) for image in images]
