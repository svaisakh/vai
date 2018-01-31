import numpy as np
import matplotlib.pyplot as plt

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


def _show_image(image, **kwargs):
    title = kwargs.pop('title', None)
    pixel_range = kwargs.pop('pixel_range', (0, 255))
    cmap = kwargs.pop('cmap', None)
    ax = kwargs.pop('ax', None)
    retain = kwargs.pop('retain', False)

    def _handle_args():
        nonlocal pixel_range, ax
        if type(image) is not np.ndarray:
            raise TypeError('image needs to be a numpy array. Found {}'.format(type(image)))
        if len(image.shape) not in (2, 3):
            raise ValueError('invalid image dimensions. Needs to be 2 or 3-D. Found {}'.format(len(image.shape)))

        if title is not None and type(title) is not str:
            raise TypeError('title needs to be None or a valid string. Found {}'.format(title))

        if type(pixel_range) not in [tuple, list, np.ndarray]:
            if type(pixel_range) is str:
                if pixel_range == 'auto':
                    pixel_range = (image.min(), image.max())
                else:
                    raise ValueError('pixel_range should be auto. Found {}'.format(pixel_range))
            else:
                raise TypeError("pixel_range needs to be a tuple, list, numpy array or 'auto'."
                                " Found {}".format(type(pixel_range)))
        elif len(pixel_range) != 2:
            raise ValueError('pixel_range needs to be of size 2 - (min, max). Found size {}'.format(len(pixel_range)))

        if ax is None:
            ax = plt.subplots()[1]

        if type(retain) is not bool:
            raise TypeError('retain must be either True or False')

    err_arg = _handle_args()
    if err_arg is not None:
        return err_arg

    ax.imshow(((image - pixel_range[0]) / (pixel_range[1] - pixel_range[0])), cmap, vmin=0, vmax=1, **kwargs)
    if title is not None:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid('off')

    if not retain:
        plt.show()
