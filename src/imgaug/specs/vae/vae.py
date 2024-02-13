import torch
from PIL import Image
from imgaug.image import SmallestMaxSize
from torchvision import transforms


class VaeTransform:
    def __init__(self, size, random_crop=False, resample=None):
        self.size = size

        resize = SmallestMaxSize(size, resample=resample)
        if not random_crop:
            croper = transforms.CenterCrop(size=(size, size))
        else:
            croper = transforms.RandomCrop(size=(size, size))

        self.__imprec = transforms.Compose([
            resize, 
            croper,
            transforms.ToTensor()])

    def __call__(self, image: Image):
        image = self.__imprec(image)
        image = image * 2 - 1.0
        return image

    @staticmethod 
    def decode(latent: torch.Tensor):
        # latent: [N, C, H, W]
        image = (latent/2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images