import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, PILToTensor, ToPILImage, InterpolationMode
import os
import cv2
import numpy as np
import pandas as pd

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from albumentations.augmentations.transforms import RandomGamma

import segmentation_models_pytorch as smp
import argparse

# Set device type
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu') 
class UnetDataClass(Dataset):
    def __init__(self, image_dir, transform, mask_dir = [],  mode = "train"):
        super(UnetDataClass, self).__init__()
        self.mode = mode
        if mode == "train":
            self.train_path = image_dir
            self.train_mask_path = mask_dir
            self.len = len(self.train_path)
            self.train_transform = transform
        elif mode == "valid":
            self.val_path = image_dir
            self.val_mask_path = mask_dir
            self.len = len(self.val_path)
            self.val_transform = transform
        elif mode == "test":
            self.test_path = image_dir
            self.len = len(self.test_path)
            self.test_transform = transform

            
    def read_mask(self, mask_path):
        image = cv2.imread(mask_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = cv2.resize(image, (256,256))
        # lower boundary RED color range values; Hue (0 - 10)
        lower1 = np.array([0, 100, 20])
        upper1 = np.array([10, 255, 255])
        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([160, 100, 20])
        upper2 = np.array([179, 255, 255])
        lower_mask = cv2.inRange(image, lower1, upper1)
        upper_mask = cv2.inRange(image, lower2, upper2)

        red_mask = lower_mask + upper_mask
        red_mask[red_mask != 0] = 1

        # boundary GREEN color range values; Hue (36 - 70)
        green_mask = cv2.inRange(image, (36, 25, 25), (70, 255, 255))
        green_mask[green_mask != 0] = 2

        full_mask = cv2.bitwise_or(red_mask, green_mask)
        full_mask = full_mask.astype(np.uint8)
        return full_mask


    def __getitem__(self, index: int):
        if self.mode == "train":
            image = cv2.imread(self.train_path[index])
            image = cv2.resize(image, (256,256))
            mask = self.read_mask(self.train_mask_path[index])
            return self.train_transform(image=image, mask=mask)
        elif self.mode == "valid":
            image = cv2.imread(self.val_path[index])
            image = cv2.resize(image, (256,256))
            mask = self.read_mask(self.val_mask_path[index])
            return self.val_transform(image=image, mask=mask)
        elif self.mode == "test":
            image = cv2.imread(self.test_path[index])
            H, W, _ = image.shape
            image = cv2.resize(image, (256,256))
            image = self.test_transform(image=image)
            
            file_name = self.test_path[index].split('/')[-1].split('.')[0]
            return  image, file_name,H, W
        
    def __len__(self):
        return self.len
    
# Load the model
model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3
)
model.to(device)

parser = argparse.ArgumentParser(description='Polyp Segmentation')
parser.add_argument('--checkpoint', type=str, help='Path to the model checkpoint')
parser.add_argument('--test_dir', type=str, help='Directory path to test images')
parser.add_argument('--predict_dir', type=str, help='Directory path to save output masks')
args = parser.parse_args()

# Load the checkpoint
checkpoint = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(checkpoint['model'])


# Define the transformation for validation images
normal_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

test_path = []
tests_path = args.test_dir
for root, dirs, files in os.walk(tests_path):
    for file in files:
        path = os.path.join(root,file)
        test_path.append(path)

unet_test_dataset = UnetDataClass(test_path, normal_transform, mode = "test")
test_dataloader = DataLoader(unet_test_dataset, batch_size=8, shuffle=False)
# Process test images
model.eval()
def mask2rgb(mask):
    color_dict = {0: torch.tensor([0, 0, 0]),
                  1: torch.tensor([1, 0, 0]),
                  2: torch.tensor([0, 1, 0])}
    output = torch.zeros((mask.shape[0], mask.shape[1], 3)).long()
    for k in color_dict.keys():
        output[mask.long() == k] = color_dict[k]
    return output.to(mask.device)

if not os.path.isdir("/kaggle/working/predicted_masks"):
    os.mkdir("/kaggle/working/predicted_masks")
for _, (img, path, H, W) in enumerate(test_dataloader):
    a = path
    b = img['image']
    h = H
    w = W
    
    with torch.no_grad():
        predicted_mask = model(b)
    for i in range(len(a)):
        image_id = a[i].split('/')[-1].split('.')[0]
        filename = image_id + ".png"
        argmax = torch.argmax(predicted_mask[i], 0)
        one_hot = mask2rgb(argmax).float().permute(2, 0, 1)
        mask2img = Resize((H[i].item(), W[i].item()), interpolation=InterpolationMode.NEAREST)(ToPILImage()(one_hot))
#         mask2img = Resize((h[i].item(), w[i].item()), interpolation=InterpolationMode.NEAREST)(ToPILImage()(F.one_hot(torch.argmax(predicted_mask[i], 0)).permute(2, 0, 1).float()))
        mask2img.save(os.path.join(args.predict_dir, filename))

# Produce Output
def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

def rle_encode_one_mask(mask):
    pixels = mask.flatten()
    pixels[pixels > 0] = 255
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle_to_string(rle)

def mask2string(dir):
    ## mask --> string
    strings = []
    ids = []
    ws, hs = [[] for i in range(2)]
    for image_id in os.listdir(dir):
        id = image_id.split('.')[0]
        path = os.path.join(dir, image_id)
        print(path)
        img = cv2.imread(path)[:,:,::-1]
        h, w = img.shape[0], img.shape[1]
        for channel in range(2):
            ws.append(w)
            hs.append(h)
            ids.append(f'{id}_{channel}')
            string = rle_encode_one_mask(img[:,:,channel])
            strings.append(string)
    r = {
        'ids': ids,
        'strings': strings,
    }
    return r


MASK_DIR_PATH = args.predict_dir 
dir = MASK_DIR_PATH
res = mask2string(dir)
df = pd.DataFrame(columns=['Id', 'Expected'])
df['Id'] = res['ids']
df['Expected'] = res['strings']
df.to_csv(r'submission.csv', index=False)
