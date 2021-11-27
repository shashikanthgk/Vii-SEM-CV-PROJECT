import torch
import os
import cv2
from preprocess import one_hot_encode 
import albumentations as album
class BuildingsDataset(torch.utils.data.Dataset):

    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            class_rgb_values = None, 
            augmentation = None, 
            preprocessing = None,
    ):
        
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        
        # Read images and masks
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)
        
        # One-hot-encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')
        
        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image = image, mask = mask)
            image, mask = sample['image'], sample['mask']
        
        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image = image, mask = mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        # Return length 
        return len(self.image_paths)

def get_training_augmentation():
    train_transform = [    
        album.RandomCrop(height = 256, width = 256, always_apply = True),
        album.OneOf(
            [
                album.HorizontalFlip(p = 1),
                album.VerticalFlip(p = 1),
                album.RandomRotate90(p = 1),
            ],
            p = 0.75,
        ),
    ]
    return album.Compose(train_transform)

def get_validation_augmentation():   
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        album.PadIfNeeded(min_height = 1536, min_width = 1536, always_apply = True, border_mode = 0),
    ]
    return album.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')
def get_preprocessing(preprocessing_fn=None):
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image = preprocessing_fn))
    _transform.append(album.Lambda(image = to_tensor, mask = to_tensor))
        
    return album.Compose(_transform)