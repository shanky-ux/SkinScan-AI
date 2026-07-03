import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

def validate_image(image):
    """
    Validate if the uploaded image is valid and processable
    """
    try:
        if image is None:
            return False
        
        # Check if image has valid dimensions
        if image.size[0] < 32 or image.size[1] < 32:
            return False
        
        # Check if image has valid mode
        if image.mode not in ['RGB', 'RGBA', 'L']:
            return False
        
        return True
    except Exception:
        return False

def preprocess_image(image):
    """
    Advanced preprocessing for maximum model accuracy
    """
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply image enhancement for better quality
        enhanced_image = enhance_image_quality(image)
        
        # Define optimized preprocessing pipeline for maximum accuracy
        transform = transforms.Compose([
            # Resize with high quality interpolation
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
            # Center crop to exact EfficientNet input size
            transforms.CenterCrop(224),
            # Convert to tensor
            transforms.ToTensor(),
            # Normalize with EfficientNet pretrained parameters
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Apply transforms
        processed_image = transform(enhanced_image)
        
        return processed_image
        
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

def preprocess_image_with_tta(image):
    """
    Test Time Augmentation (TTA) for even higher accuracy
    """
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        enhanced_image = enhance_image_quality(image)
        
        # Multiple augmentations for TTA
        transforms_list = [
            # Original
            transforms.Compose([
                transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # Slight rotation
            transforms.Compose([
                transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomRotation(5),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # Horizontal flip
            transforms.Compose([
                transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        ]
        
        # Apply all transforms
        augmented_images = []
        for transform in transforms_list:
            augmented_images.append(transform(enhanced_image))
        
        return torch.stack(augmented_images)
        
    except Exception as e:
        raise Exception(f"Error in TTA preprocessing: {str(e)}")

def enhance_image_quality(image):
    """
    Apply basic image enhancement techniques
    """
    try:
        # Convert PIL to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Convert back to PIL
        enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        
        return enhanced_pil
        
    except Exception:
        # Return original image if enhancement fails
        return image

def resize_image_for_display(image, max_width=400):
    """
    Resize image for display purposes while maintaining aspect ratio
    """
    try:
        width, height = image.size
        
        if width > max_width:
            ratio = max_width / width
            new_height = int(height * ratio)
            image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
        
        return image
        
    except Exception:
        return image

def extract_skin_region(image):
    """
    Basic skin region extraction using color-based segmentation
    This is a simplified approach for demonstration
    """
    try:
        # Convert to HSV for better skin color detection
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2HSV)
        
        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask for skin regions
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply mask to original image
        result = cv2.bitwise_and(opencv_image, opencv_image, mask=mask)
        
        # Convert back to PIL
        result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        
        return result_pil
        
    except Exception:
        # Return original image if skin extraction fails
        return image

def get_image_statistics(image):
    """
    Get basic statistics about the image for quality assessment
    """
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Calculate basic statistics
        stats = {
            'width': image.size[0],
            'height': image.size[1],
            'channels': len(img_array.shape),
            'mean_brightness': np.mean(img_array),
            'std_brightness': np.std(img_array),
            'min_value': np.min(img_array),
            'max_value': np.max(img_array)
        }
        
        # Assess image quality
        if stats['std_brightness'] < 20:
            stats['quality_assessment'] = "Low contrast - image may be too dark or bright"
        elif stats['mean_brightness'] < 50:
            stats['quality_assessment'] = "Image appears too dark"
        elif stats['mean_brightness'] > 200:
            stats['quality_assessment'] = "Image appears too bright"
        else:
            stats['quality_assessment'] = "Good image quality"
        
        return stats
        
    except Exception:
        return {
            'width': 0,
            'height': 0,
            'quality_assessment': "Unable to assess image quality"
        }
