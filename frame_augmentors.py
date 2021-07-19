import imgaug as ia
import imgaug.augmenters as iaa
import torchvision.transforms.functional as TF
import torch

class MaskRandomCrop:
    """Rotate by one of the given angles."""

    def __init__(self, crop_size):
        # size of square crop
        self.crop_size = crop_size

    def __call__(self, sample):
        im = sample['image']
        mask = sample['mask']

        im_height = im.shape[0]
        im_width = im.shape[1]
        top = np.random.randint(im_height - self.crop_size)
        left = np.random.randint(im_width - self.crop_size)

        im_crop = im[top:top+self.crop_size, left:left+self.crop_size]
        mask_crop = mask[top:top+self.crop_size, left:left+self.crop_size]
        
        sample['image'] = im_crop
        sample['mask'] = mask_crop

        return sample
    
class MaskImgAug:
    
    def __init__(self):
        # size of square crop
        self.transform = iaa.Sequential([
#             iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)),
            iaa.Sometimes(0.4, iaa.blur.GaussianBlur(sigma=(0.0, 6.0))),
            iaa.Sometimes(0.4, iaa.AdditivePoissonNoise((0, 30)))
        ])

    def __call__(self, sample):
        im = sample['image']
        im_transform = self.transform(image=np.array(im))
        sample['image'] = im_transform
        
        return sample
    
class Mask2dMultiplyAndAddToBrightness():
    def __init__(self, add, multiply):
        """Add and multiply image values by given amount randomly within range.
        
        Expects image to be between 0 and 255.
        
        Args:
            add: either number of tuple, if tuple randomly choose from range
            multiply: either number ot tuple, if tuple randomly choose from range
        """
        self.add = add
        self.multiply = multiply
        self.rng = np.random.default_rng()
        
    def __call__(self, sample):
        im = sample['image'].astype(np.float64)
        if isinstance(self.add, tuple):
            add_val = self.rng.uniform(self.add[0], self.add[1])
        else:
            add_val = self.add
        if isinstance(self.multiply, tuple):
            multiply_val = self.rng.uniform(self.multiply[0], self.multiply[1])
        else:
            multiply_val = self.multiply
        im += add_val
        im *= multiply_val
        im = np.maximum(im, 0)
        im = np.minimum(im, 255)
        
        sample['image'] = im.astype(np.uint8)
        
        return sample
    
    
class MaskToTensor:
    def __call__(self, sample):
        im = sample['image']
        im_tensor = torch.from_numpy(im).float() / 255
        im_tensor = torch.unsqueeze(im_tensor, 0)
        sample['image'] = im_tensor
        
        if 'mask' in sample.keys():
            mask = sample['mask']
            mask_tensor = torch.from_numpy(np.array(mask, np.int64, copy=False))
            mask_tensor = mask_tensor // 255
            sample['mask'] = mask_tensor
        
        return sample
    
class Mask3dto2d:
    """ Expects images of size NxMxC"""
    def __init__(self, channel_to_use):
        self.channel_to_use = channel_to_use
    
    def __call__(self, sample):
        im = sample['image']
        sample['image'] = im[..., self.channel_to_use]
       
        return sample
    
class MaskCompose:
    def __init__(self, transform_list):
        """Chain together transforms in transform.
         Expect transform on both image and mask in dict
         
         Args:
             transform_list (list): list of custom augmentations
         """
        self.transform_list = transform_list
    
    def __call__(self, sample):
        for transform in self.transform_list:
            sample  = transform(sample)
        return sample
    
class MaskNormalize:
    def __init__(self, mean, std):
        """Normalize image, leave mask unchanged"""
        self.mean = mean
        self.std = std
        
    def __call__(self, sample):
        im = sample['image']
        im_norm = TF.normalize(im, self.mean, self.std)
        
        sample['image'] = im_norm
        return sample
    
class MaskContrast:
    def __init__(self, contrast_factor):
        """Change image contrast, leave mask unchanged
        
        Args:
            contrast_factor: single value or tuple"""
        self.contrast_factor = contrast_factor
        self.rng = np.random.default_rng()

    def __call__(self, sample):
        im = sample['image']
        if isinstance(self.contrast_factor, tuple):
            contrast_factor = self.rng.uniform(self.contrast_factor[0], self.contrast_factor[1])
        else:
            contrast_factor = self.contrast_factor
        im_height, im_width = im.shape
        aprox_mean = np.mean([im[0,0], im[-1, 0], im[0, -1], im[-1, -1], 
                              im[im_height//2, im_width//2], im[-im_height//2, -im_width//2]])
        im_contrast = aprox_mean + contrast_factor * (im - aprox_mean)
        sample['image'] = im_contrast.astype(np.uint8)
        return sample
    
class AddDim:
    """Add dimension in the given postion."""
    
    def __init__(self, new_dim):
        self.new_dim = new_dim
    
    def __call__(self, sample):
        sample['image'] = torch.unsqueeze(sample['image'], self.new_dim)
        return sample
    
class ToFloat:
    def __init__(self):
        pass
    
    def __call__(self, sample):
        sample['image'] = sample['image'].float()
        return sample