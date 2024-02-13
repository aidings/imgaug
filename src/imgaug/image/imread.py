import os
from PIL import Image
import numpy as np
from pathlib import Path
import imageio.v3 as iio
from io import BytesIO
from loguru import logger


def to_pil(img_buf):
    if isinstance(img_buf, bytes):
        image = iio.imread(img_buf)
        image = Image.fromarray(image)
    elif isinstance(img_buf, (Path, str)) and Path(img_buf).exists():
        image = Image.open(img_buf)
        image = image.convert('RGB')
    elif isinstance(img_buf, np.ndarray):
        image = Image.fromarray(img_buf)
    elif isinstance(img_buf, Image.Image):
        image = img_buf
    else:
        image = None
        logger.error(f'Input Image error, please check out {img_buf}')

    return image


def to_npy(img_buf):
    if isinstance(img_buf, bytes):
        image = iio.imread(img_buf)
    elif isinstance(img_buf, str) and os.path.isfile(img_buf):
        image = iio.imread(img_buf)
    elif isinstance(img_buf, str) and img_buf.startswith('http'):
        image = iio.imread(img_buf)
    elif isinstance(img_buf, np.ndarray):
        image = img_buf
    elif isinstance(img_buf, Image.Image):
        image = np.asarray(img_buf)
    else:
        image = None
        logger.error('Not support this type image buffer(byte, str, np.ndarry, PIL.Image)')
    return image


def image2bytes(img_buf):
    if isinstance(img_buf, (Path, str)):
        return open(img_buf, 'rb').read()
    elif isinstance(img_buf, Image.Image):
        bytesIO = BytesIO()
        img_buf.save(bytesIO, format='PNG')
        return bytesIO.getvalue()
    else:
        logger.error('Error: Not support this type image buffer(byte, str, PIL.Image)')
        return None
