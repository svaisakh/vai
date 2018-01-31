import numpy as np


def _colorize_images(images):
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
