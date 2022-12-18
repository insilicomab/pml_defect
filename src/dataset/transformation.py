from torchvision import transforms
from omegaconf import DictConfig


class Transforms():

    def __init__(self, cfg: DictConfig):
        
        self.data_transform = {
            'train': transforms.Compose([
                transforms.Resize(
                    (cfg.train_transform.resize.image_size, cfg.train_transform.resize.image_size)
                    ),
                transforms.RandomHorizontalFlip(p=cfg.train_transform.random_horizontal_flip.p),
                transforms.RandomVerticalFlip(p=cfg.train_transform.random_vertical_flip.p),
                transforms.RandomRotation(degrees=cfg.train_transform.random_rotation.degrees),
                transforms.RandomAffine(
                    degrees=cfg.train_transform.random_affine.degrees,
                    translate=cfg.train_transform.random_affine.translate,
                    scale=cfg.train_transform.random_affine.scale,
                    shear=cfg.train_transform.random_affine.shear,
                    ),
                transforms.ColorJitter(
                    brightness=cfg.train_transform.color_jitter.brightness,
                    contrast=cfg.train_transform.color_jitter.contrast,
                    saturation=cfg.train_transform.color_jitter.saturation,
                    hue=cfg.train_transform.color_jitter.hue
                    ),
                transforms.ToTensor(),
                transforms.Normalize(
                    cfg.train_transform.normalize.mean,
                    cfg.train_transform.normalize.std
                    ),
                ]),
            'val': transforms.Compose([
                transforms.Resize(
                    (cfg.test_transform.resize.image_size, cfg.test_transform.resize.image_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    cfg.test_transform.normalize.mean,
                    cfg.test_transform.normalize.std
                    ),
                ]),
            'test': transforms.Compose([
                transforms.Resize(
                    (cfg.test_transform.resize.image_size, cfg.test_transform.resize.image_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    cfg.test_transform.normalize.mean,
                    cfg.test_transform.normalize.std
                    ),
                ]),
        }
    
    def __call__(self, phase, img):
        """
        Parameters
        ----------
        phase : 'train' or 'val' or 'test'
        """
        return self.data_transform[phase](img)


class TestTransforms():

    def __init__(self, image_size):
        
        self.data_transform = {
            'test': transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                    ),
                ]),
        }
    
    def __call__(self, phase, img):
        return self.data_transform[phase](img)