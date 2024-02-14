import PIL
import numbers
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from skimage.color import rgb2gray
from skimage.feature import canny
from PIL.ImageOps import expand as pad
import random


    

class Normalize(object):
    def __init__(self, mean, std) -> None:
        self.mean = np.asarray(mean).reshape(-1, 1, 1).astype(np.float32)
        self.std = np.asarray(std).reshape(-1, 1, 1).astype(np.float32)
    
    def __call__(self, image):
        image = image.astype(np.float32)
        image = (image - self.mean) / self.std

        return image


class ToNumpy:
    def __init__(self, norm=False) -> None:
        if norm:
            self.norm = lambda x: x.astype(np.float32) / 255.0
        else:
            self.norm = lambda x: x

    def __call__(self, image: Image.Image):
        image = np.asarray(image)
        image = self.norm(image)
        return image


class CenterCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError('If size is a single number, it must be positive')
            size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError('If size is a sequence, it must be of len 2.')
        self.size = size
    
    def __call__(self, image: Image.Image):
        crop_h, crop_w = self.size
        im_w, im_h = image.size

        if crop_w > im_w or crop_h > im_h:
            error_msg = ('Initial image size should be larger then' +
                         'cropped size but got cropped sizes : ' +
                         '({w}, {h}) while initial image is ({im_w}, ' +
                         '{im_h})'.format(im_w=im_w, im_h=im_h, w=crop_w,
                                          h=crop_h))
            raise ValueError(error_msg)

        w1 = int(round((im_w - crop_w) / 2.))
        h1 = int(round((im_h - crop_h) / 2.))

        return image.crop((w1, h1, w1 + crop_w, h1 + crop_h))


class Canny:
    def __init__(self, sigma=1, low_threshold=None, high_threshold=None, use_quantiles=False) -> None:
        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.use_quantiles = use_quantiles

    def __call__(self, image):
        gray = rgb2gray(image)
        image = canny(gray, sigma=self.sigma, low_threshold=self.low_threshold, high_threshold=self.high_threshold, use_quantiles=self.use_quantiles)
        image = image.astype(np.uint8) * 255
        return PIL.Image.fromarray(image) 


def _round(number):
    """Unified rounding in all python versions."""
    if abs(round(number) - number) == 0.5:
        return int(2.0 * round(number / 2.0))
    return int(round(number))


class LongestMaxSize:
    def __init__(self, max_size, resample=None):
        self.max_size = max_size
        self.resample = resample
    
    def __call__(self, image):
        width, height = image.size

        scale = self.max_size / float(max(width, height))

        if scale != 1.0:
            nw, nh = [_round(dim * scale) for dim in (width, height)]
            image = image.resize((nw, nh), resample=self.resample)
        
        return image


class SmallestMaxSize:
    """ Rescale the input PIL Image to be the smaller edge of max_size.
    """
    def __init__(self, max_size, resample=None):
        self.max_size = max_size
        self.resample = resample
    
    def __call__(self, image):
        width, height = image.size

        scale = self.max_size / float(min(width, height))

        if scale != 1.0:
            nw, nh = [_round(dim * scale) for dim in (width, height)]
            image = image.resize((nw, nh), resample=self.resample)
        
        return image


class ResizeWithLongSide:
    def __init__(self, size, resample=Image.BILINEAR):
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.resample = resample

    def __call__(self, image, size):
        assert isinstance(image, Image.Image), f"Only support PIL.Image, but input({type(image)})"
        w, h = image.size
        wr = size[0] / w
        hr = size[1] / h
        if w > h:
            new_w = size[0]
            new_h = int(wr * h)
            scale = wr
        elif w < h:
            new_h = size[1]
            new_w = int(hr * w)
            scale = hr
        else:
            new_w = size[0]
            new_h = size[1]
            scale = hr

        image = image.resize((new_w, new_h), resample=self.resample)

        return image, scale


class PaddingAndResize:
    def __init__(self, size, is_center=True, resampler=Image.BILINEAR, fill=0) -> None:
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        elif isinstance(size, tuple):
            size = (size[0], size[1])
        else:
            raise RuntimeError(f'Not Support this kind of input({size})')
        self.size = size
        self.is_center = is_center
        self.resampler = resampler
        self.fill = fill
        self.resize_with_longside = ResizeWithLongSide(size, resampler)
    
    def __call__(self, image):
        image, rate = self.resize_with_longside(image)
        ow, oh = image.size
        iw, ih = self.size
        wpad = (iw - ow) // 2
        hpad = (ih - oh) // 2
        if self.is_center:
            border = (wpad, hpad, wpad, hpad)
            image = pad(image, border, fill=self.fill)
            box = (wpad, hpad, wpad+ow, hpad+oh)
        else:
            border = (0, 0, wpad, hpad)
            image = pad(image, border, fill=self.fill)
            box = (0, 0, ow, oh)
        
        return image, {'rate': rate, 'box': box}


class InvPaddingAndResize:
    def __init__(self, resampler=Image.BILINEAR) -> None:
        self.resampler = resampler
    
    def __call__(self, image, rate, box):
        roi = image.crop(box)
        w, h = roi.size
        ow, oh = int(w / rate), int(h / rate)
        image = roi.resize((ow, oh), resample=self.resampler)
        return image


class Compose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.PILToTensor(),
        >>>     transforms.ConvertImageDtype(torch.float),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    """

    def __init__(self, transforms, with_dict=False):
        self.transforms = transforms
        self.__func = self.__call_dict if with_dict else self.__call_norm
    
    def __call_norm(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __call_dict(self, img):
        rdict = {}
        for t in self.transforms:
            img = t(img)
            if isinstance(img, tuple) and len(img) == 2 and isinstance(img[1], dict):
                rdict.update({type(t).__name__:img[1]})
                img = img[0]
        return img, rdict

    def __call__(self, img):
        return self.__func(img)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class ShearX(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
            Image.BICUBIC, fillcolor=self.fillcolor)


class ShearY(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
            Image.BICUBIC, fillcolor=self.fillcolor)


class TranslateX(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, 0, magnitude * x.size[0] * random.choice([-1, 1]), 0, 1, 0),
            fillcolor=self.fillcolor)


class TranslateY(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * x.size[1] * random.choice([-1, 1])),
            fillcolor=self.fillcolor)


class Rotate(object):
    # from https://stackoverflow.com/questions/
    # 5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
    def __call__(self, x, magnitude):
        rot = x.convert("RGBA").rotate(magnitude * random.choice([-1, 1]))
        return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(x.mode)


class Color(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Color(x).enhance(1 + magnitude * random.choice([-1, 1]))


class Posterize(object):
    def __call__(self, x, magnitude):
        return ImageOps.posterize(x, magnitude)


class Solarize(object):
    def __call__(self, x, magnitude):
        return ImageOps.solarize(x, magnitude)


class Contrast(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Contrast(x).enhance(1 + magnitude * random.choice([-1, 1]))


class Sharpness(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Sharpness(x).enhance(1 + magnitude * random.choice([-1, 1]))


class Brightness(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Brightness(x).enhance(1 + magnitude * random.choice([-1, 1]))


class AutoContrast(object):
    def __call__(self, x, magnitude):
        return ImageOps.autocontrast(x)


class Equalize(object):
    def __call__(self, x, magnitude):
        return ImageOps.equalize(x)


class Invert(object):
    def __call__(self, x, magnitude):
        return ImageOps.invert(x)


class CenterRandomScale(object):
    def __init__(self, p=0.5, scale=(1.0, 1.25)):
        self.p = p
        self.scale = scale
    
    def __call__(self, xyxy, imsize):
        if random.random() >= self.p:
            scale = random.uniform(*self.scale)
            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]

            new_w = int(w * scale / 2)
            new_h = int(h * scale / 2)
            x1 = max(cx - new_w, 0)
            y1 = max(cy - new_h, 0)
            x2 = min(cx + new_w, imsize[0])
            y2 = min(cy + new_h, imsize[1])
        else:
            x1, y1, x2, y2 = xyxy
        
        return x1, y1, x2, y2


class IncludeResize:
    def __init__(self, size, resample=Image.BILINEAR):
        self.size = size 
        self.resample = resample
    
    def __call__(self, image):
        src_w, src_h = image.size
        dst_w, dst_h = self.size

        # resize width
        wrate = dst_w / src_w
        hrate = dst_h / src_h

        if wrate > hrate:
            # padding height
            new_w = dst_w
            new_h = int(src_h * wrate)

            dh = (new_h - dst_h) // 2
            left, right = 0, dst_w
            top, bottom = dh, dst_h + dh
        else:
            # padding width
            new_w = int(src_w * hrate)
            new_h = src_h
            dw = (new_w - dst_w) // 2
            left, right = dw, dst_w + dw
            top, bottom = 0, dst_h
        image = image.resize((new_w, new_h), resample=self.resample)
        image = image.crop((left, top, right, bottom))

        return image