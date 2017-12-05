'''
Some transforms taken from
https://github.com/ZijunDeng/pytorch-semantic-segmentation
'''

from numpy import array, int32, random, asarray
from numpy import linspace, meshgrid, dstack, vstack, sin
from numpy.random import normal
from skimage.transform import estimate_transform, warp
from PIL import Image, ImageOps
import torch
import numbers

class MaskToTensor(object):
    def __call__(self, img):
        imgarray = array(img, dtype=int32)
        return torch.from_numpy(imgarray/imgarray.max()).long()

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img, mask
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)

class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        r = torch.rand(1).numpy()
        if r < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

class RandomVerticallyFlip(object):
    def __call__(self, img, mask):
        r = torch.rand(1).numpy()
        if r < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
        return img, mask

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = torch.rand(1).numpy() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        r = torch.rand(1).numpy()
        x1 = int(r*(w - tw))
        r = torch.rand(1).numpy()
        y1 = int(r*(h - th))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))

class RandomWarp(object):
    def __init__(self, controlpoints, scale):
        if isinstance(controlpoints, numbers.Number):
            self.controlpoints = (int(controlpoints), int(controlpoints))
        else:
            self.controlpoints = controlpoints
        self.scale = scale

    def __call__(self, img, mask):

        cols = img.size[1]
        rows = img.size[0]

        src_cols = linspace(0, cols, self.controlpoints[1])
        src_rows = linspace(0, rows, self.controlpoints[0])
        src_rows, src_cols = meshgrid(src_rows, src_cols)
        src = dstack([src_cols.flat, src_rows.flat])[0]

        dst_rows = src[:, 1] + self.scale*torch.randn(src[:, 1].shape).numpy()
        dst_cols = src[:, 0] + self.scale*torch.randn(src[:, 1].shape).numpy()
        dst = vstack([dst_cols, dst_rows]).T

        tform = estimate_transform('piecewise-affine', src, dst)
        warped_img = warp(asarray(img), tform, output_shape=img.size)
        warped_mask = warp(asarray(mask), tform, output_shape=mask.size)
        return Image.fromarray((255*warped_img).astype('uint8')), Image.fromarray((255*warped_mask).astype('uint8'))
