from PIL import Image
from torchvision import transforms
from imgaug.math import _auto_size


class FineGrainedTransform:
    def __init__(self, size, rate=0.875, state='train'):
        resize = _auto_size(size)
        if state == 'train':
            self.trans = transforms.Compose([
                transforms.Resize(size=(int(resize[0] / rate), int(resize[1] / rate))),
                transforms.RandomCrop(resize),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.126, saturation=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.trans = transforms.Compose([
                transforms.Resize(size=(int(resize[0] / rate), int(resize[1] / rate))),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __call__(self, image: Image.Image):
        return self.trans(image)