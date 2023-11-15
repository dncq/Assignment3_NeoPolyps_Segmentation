import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from albumentations.augmentations.transforms import RandomGamma

class TrainTransform:
    def __init__(self):
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomGamma (gamma_limit=(70, 130), eps=None, always_apply=False, p=0.2),
            A.RGBShift(p=0.3, r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
            A.OneOf([A.Blur(), A.GaussianBlur(), A.GlassBlur(), A.MotionBlur(), A.GaussNoise(), A.Sharpen(), A.MedianBlur(), A.MultiplicativeNoise()]),
            A.Cutout(p=0.2, max_h_size=35, max_w_size=35, fill_value=255),
            A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.15, brightness_coeff=1.5, p=0.09),
            A.RandomShadow(p=0.1),
            A.ShiftScaleRotate(p=0.45, border_mode=cv2.BORDER_CONSTANT, shift_limit=0.15, scale_limit=0.15),
            A.RandomCrop(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __call__(self, img, mask):
        return self.transform(image=img, mask=mask)


class ValTransform:
    def __init__(self):
        self.transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    def __call__(self, img, mask):
        return self.transform(image=img, mask=mask)


class TestTransform:
    def __init__(self):
        self.transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    def __call__(self, img):
        return self.transform(image=img)['image']