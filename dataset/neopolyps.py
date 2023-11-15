# Loading the dataset
trainsize = 384

class BKpolypDataset(torch.utils.data.Dataset):
    def __init__(self, dir="path/to/data", transform=None):
        self.img_path_lst = []
        self.dir = dir
        self.transform = transform
        self.img_path_lst = glob.glob("{}/train/train/*".format(self.dir))

    def __len__(self):
        return len(self.img_path_lst)

    def read_mask(self, mask_path):
        image = cv2.imread(mask_path)
        image = cv2.resize(image, (trainsize, trainsize))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # lower boundary RED color range values; Hue (0 - 10)
        lower1 = np.array([0, 100, 20])
        upper1 = np.array([10, 255, 255])
        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([160,100,20])
        upper2 = np.array([179,255,255])
        lower_mask = cv2.inRange(image, lower1, upper1)
        upper_mask = cv2.inRange(image, lower2, upper2)
        red_mask = lower_mask + upper_mask;
        red_mask[red_mask != 0] = 2
        # boundary RED color range values; Hue (36 - 70)
        green_mask = cv2.inRange(image, (36, 25, 25), (70, 255,255))
        green_mask[green_mask != 0] = 1
        full_mask = cv2.bitwise_or(red_mask, green_mask)
        full_mask = full_mask.astype(np.uint8)
        return full_mask

    def __getitem__(self, idx):
        img_path = self.img_path_lst[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (trainsize, trainsize))
        label_path = img_path.replace("train", "train_gt")
        label = self.read_mask(label_path)

        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]
            label = transformed["mask"]
        return image, label