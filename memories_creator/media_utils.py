"""
Media utilities for Apple Memories Creator
Handles format conversion, EXIF data, and image processing
"""

import os
import subprocess
from PIL import Image, ExifTags, ImageEnhance
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import tempfile
import numpy as np


class MediaConverter:
    """Handles conversion of various image formats"""
    
    SUPPORTED_EXTENSIONS = {
        'standard': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'],
        'apple': ['.heic', '.heif'],
        'raw': ['.dng', '.cr2', '.nef', '.arw']
    }
    
    @staticmethod
    def is_supported(filepath: str) -> bool:
        """Check if file format is supported"""
        ext = Path(filepath).suffix.lower()
        all_supported = (
            MediaConverter.SUPPORTED_EXTENSIONS['standard'] +
            MediaConverter.SUPPORTED_EXTENSIONS['apple'] +
            MediaConverter.SUPPORTED_EXTENSIONS['raw']
        )
        return ext in all_supported
    
    @staticmethod
    def needs_conversion(filepath: str) -> bool:
        """Check if file needs conversion to standard format"""
        ext = Path(filepath).suffix.lower()
        return ext in (
            MediaConverter.SUPPORTED_EXTENSIONS['apple'] +
            MediaConverter.SUPPORTED_EXTENSIONS['raw']
        )
    
    @staticmethod
    def convert_to_jpeg(filepath: str, output_dir: str = None) -> Optional[str]:
        """
        Convert image to JPEG format
        
        Args:
            filepath: Source file path
            output_dir: Output directory (uses temp if None)
            
        Returns:
            Path to converted file or None if failed
        """
        try:
            # Try direct PIL conversion first
            img = Image.open(filepath)
            
            # Create output path
            if output_dir is None:
                output_dir = tempfile.gettempdir()
            
            filename = Path(filepath).stem + '_converted.jpg'
            output_path = os.path.join(output_dir, filename)
            
            # Convert to RGB if needed
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save as JPEG
            img.save(output_path, 'JPEG', quality=95, optimize=True)
            return output_path
            
        except Exception as e:
            # Fallback to sips (macOS built-in converter)
            if os.uname().sysname == 'Darwin':
                try:
                    output_path = os.path.join(
                        output_dir or tempfile.gettempdir(),
                        Path(filepath).stem + '_converted.jpg'
                    )
                    subprocess.run([
                        'sips',
                        '-s', 'format', 'jpeg',
                        filepath,
                        '--out', output_path
                    ], check=True, capture_output=True)
                    return output_path
                except:
                    pass
            
            print(f"⚠️  Warning: Could not convert {filepath}: {e}")
            return None


class ExifParser:
    """Handles EXIF data extraction and orientation correction"""
    
    @staticmethod
    def get_date_taken(filepath: str) -> Optional[datetime]:
        """Extract date taken from EXIF data"""
        try:
            img = Image.open(filepath)
            exif = img._getexif()
            
            if exif:
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if tag == 'DateTimeOriginal':
                        return datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
        except:
            pass
        
        # Fallback to file modification time
        try:
            stat = os.stat(filepath)
            return datetime.fromtimestamp(stat.st_mtime)
        except:
            return None
    
    @staticmethod
    def get_orientation(filepath: str) -> int:
        """
        Get EXIF orientation tag
        
        Returns:
            1-8 orientation value (1 = normal)
        """
        try:
            img = Image.open(filepath)
            exif = img._getexif()
            
            if exif:
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if tag == 'Orientation':
                        return value
        except:
            pass
        
        return 1  # Normal orientation
    
    @staticmethod
    def auto_rotate(img: Image.Image) -> Image.Image:
        """
        Auto-rotate image based on EXIF orientation
        
        Args:
            img: PIL Image
            
        Returns:
            Rotated PIL Image
        """
        try:
            exif = img._getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if tag == 'Orientation':
                        if value == 3:
                            img = img.rotate(180, expand=True)
                        elif value == 6:
                            img = img.rotate(270, expand=True)
                        elif value == 8:
                            img = img.rotate(90, expand=True)
                        break
        except:
            pass
        
        return img


class ImageProcessor:
    """Process images with rotations, enhancements, and transformations"""
    
    @staticmethod
    def load_and_prepare_image(filepath: str, custom_rotation: int = 0, 
                              enhance_old: bool = False, is_old: bool = False,
                              enhance_strength: float = 0.03) -> Optional[Image.Image]:
        """
        Load image with all transformations applied for consistent processing
        
        Args:
            filepath: Path to image file
            custom_rotation: Custom rotation angle in degrees (0, 90, 180, 270)
            enhance_old: Whether to apply old photo enhancement
            is_old: Whether the photo is considered old (based on year)
            enhance_strength: Strength of enhancement (0.0 to 1.0)
            
        Returns:
            Prepared PIL Image with all transformations applied, or None if failed
        """
        try:
            # Load image
            img = Image.open(filepath).convert('RGB')
            
            # Apply EXIF auto-rotation first (corrects camera orientation)
            img = ExifParser.auto_rotate(img)
            
            # Apply custom rotation if specified (user-defined rotation)
            if custom_rotation != 0:
                # Rotate counter-clockwise by custom_rotation degrees
                img = img.rotate(-custom_rotation, expand=True)
            
            # Enhance old photos if requested
            if enhance_old and is_old:
                img = ImageProcessor.enhance_old_photo(img, enhance_strength)
            
            return img
            
        except Exception as e:
            print(f"⚠️  Warning: Could not process {filepath}: {e}")
            return None
    
    @staticmethod
    def enhance_old_photo(img: Image.Image, strength: float = 0.03) -> Image.Image:
        """
        Gently enhance old photos by improving contrast and color
        
        Args:
            img: PIL Image to enhance
            strength: Enhancement strength (0.0 = none, 1.0 = maximum)
            
        Returns:
            Enhanced PIL Image
        """
        # Convert to numpy array for efficient processing
        img_np = np.array(img).astype(np.float32) / 255.0
        
        # Gentle contrast enhancement
        mean = img_np.mean()
        img_np = (img_np - mean) * (1.0 + strength) + mean
        
        # Gentle color enhancement (boost red and green slightly more than blue)
        img_np[:, :, 0] = np.clip(img_np[:, :, 0] * (1.0 + strength), 0, 1)        # Red channel
        img_np[:, :, 1] = np.clip(img_np[:, :, 1] * (1.0 + strength), 0, 1)        # Green channel
        img_np[:, :, 2] = np.clip(img_np[:, :, 2] * (1.0 + strength * 0.5), 0, 1)  # Blue channel (less)
        
        # Convert back to PIL Image
        img_np = np.clip(img_np, 0, 1) * 255
        return Image.fromarray(img_np.astype(np.uint8))
    
    @staticmethod
    def resize_and_crop(img: Image.Image, target_width: int, target_height: int, 
                       resize_algo: str = 'bicubic') -> Image.Image:
        """
        Resize and center crop image to target dimensions
        
        Args:
            img: PIL Image to resize
            target_width: Desired output width
            target_height: Desired output height
            resize_algo: Resize algorithm ('bilinear', 'bicubic', or 'lanczos')
            
        Returns:
            Resized and cropped PIL Image
        """
        # Select resize algorithm
        if resize_algo == 'bilinear':
            resample = Image.Resampling.BILINEAR
        elif resize_algo == 'bicubic':
            resample = Image.Resampling.BICUBIC
        else:  # lanczos
            resample = Image.Resampling.LANCZOS
        
        # Calculate scale to fill target dimensions
        w, h = img.size
        scale = max(target_width / w, target_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        img = img.resize((new_w, new_h), resample)
        
        # Center crop to target dimensions
        left = (new_w - target_width) // 2
        top = (new_h - target_height) // 2
        img = img.crop((left, top, left + target_width, top + target_height))
        
        return img


class ImageAnalyzer:
    """Analyzes images for quality, characteristics, and metadata"""
    
    @staticmethod
    def get_dimensions(filepath: str) -> Tuple[int, int]:
        """Get image dimensions in pixels"""
        try:
            img = Image.open(filepath)
            return img.size
        except:
            return (0, 0)
    
    @staticmethod
    def get_orientation_type(filepath: str) -> str:
        """
        Determine if image is vertical, horizontal, or square
        
        Returns:
            'vertical', 'horizontal', or 'square'
        """
        try:
            img = Image.open(filepath)
            # Apply auto-rotation first to get correct orientation
            img = ExifParser.auto_rotate(img)
            w, h = img.size
            
            ratio = w / h if h > 0 else 1.0
            
            if ratio > 1.1:
                return 'horizontal'
            elif ratio < 0.9:
                return 'vertical'
            else:
                return 'square'
        except:
            return 'unknown'
    
    @staticmethod
    def analyze_image(filepath: str, year_threshold: int = 2020) -> Optional[Dict[str, Any]]:
        """
        Complete image analysis including metadata extraction
        
        Args:
            filepath: Path to image file
            year_threshold: Year threshold for classifying photos as "old"
            
        Returns:
            Dictionary with complete image metadata or None if failed
        """
        try:
            date_taken = ExifParser.get_date_taken(filepath)
            year = date_taken.year if date_taken else datetime.now().year
            
            return {
                'path': filepath,
                'filename': os.path.basename(filepath),
                'year': year,
                'date_taken': date_taken,
                'orientation': ImageAnalyzer.get_orientation_type(filepath),
                'dimensions': ImageAnalyzer.get_dimensions(filepath),
                'is_old': year < year_threshold,
                'custom_rotation': 0  # Default, can be updated later
            }
        except Exception as e:
            print(f"⚠️  Warning: Could not analyze {filepath}: {e}")
            return None
