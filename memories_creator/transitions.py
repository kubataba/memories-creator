"""
Transition effects for Memories Creator
Only complex transitions that require PyTorch tensors
Simple transitions (static, pan) are handled directly in VideoProcessor
"""

import torch
import numpy as np


class TorchTransitionEngine:
    """Transition engine only for effects that need PyTorch tensors"""
    
    def __init__(self, transition_type: str, intensity: float = 0.1, prev_image: torch.Tensor = None):
        """
        Initialize transition engine
        
        Args:
            transition_type: Type of transition (ken_burns, zoom_in, zoom_out, fade_in, fade_out, fade_cross, dissolve)
            intensity: Intensity of the effect (0.0 to 1.0)
            prev_image: Previous image tensor for cross-fade effects (optional)
        """
        self.transition_type = transition_type
        self.intensity = max(0.0, min(1.0, intensity))
        self.prev_image = prev_image  # For cross-fade transitions
        
        self.transitions = {
            'ken_burns': self._ken_burns,
            'zoom_in': self._zoom_in,
            'zoom_out': self._zoom_out,
            'fade_in': self._fade_in,
            'fade_out': self._fade_out,
            'fade_cross': self._fade_cross,
            'dissolve': self._dissolve,
        }
        
        if transition_type not in self.transitions:
            raise ValueError(f"Unsupported transition type: {transition_type}. "
                           f"Use 'ken_burns', 'zoom_in', 'zoom_out', 'fade_in', 'fade_out', 'fade_cross', or 'dissolve'")
        
        self.transition_func = self.transitions[transition_type]
    
    def apply(self, img_tensor: torch.Tensor, video_w: int, video_h: int, 
              frame_progress: float, prev_tensor: torch.Tensor = None) -> torch.Tensor:
        """
        Apply transition effect to PyTorch tensor
        
        Args:
            img_tensor: Current image tensor (C, H, W)
            video_w: Target video width
            video_h: Target video height
            frame_progress: Progress through transition (0.0 to 1.0)
            prev_tensor: Previous image tensor for cross-fade (optional)
        
        Returns:
            Transformed tensor (C, video_h, video_w)
        """
        if self.transition_type in ['fade_cross', 'dissolve'] and prev_tensor is not None:
            # Ensure prev_tensor is on same device as current tensor
            if prev_tensor.device != img_tensor.device:
                prev_tensor = prev_tensor.to(img_tensor.device)
            return self.transition_func(img_tensor, video_w, video_h, frame_progress, prev_tensor)
        return self.transition_func(img_tensor, video_w, video_h, frame_progress)
    
    def _ken_burns(self, img_tensor: torch.Tensor, video_w: int, video_h: int,
                   frame_progress: float) -> torch.Tensor:
        """
        Ken Burns effect: smooth zoom in with slight pan
        
        Args:
            img_tensor: Image tensor (C, H, W)
            video_w: Target video width
            video_h: Target video height
            frame_progress: Progress through transition (0.0 to 1.0)
        
        Returns:
            Transformed tensor with Ken Burns effect
        """
        c, h, w = img_tensor.shape
        
        # Calculate zoom factor (1.0 to 1.0 + intensity)
        zoom_factor = 1.0 + (self.intensity * frame_progress * 0.7)
        
        # Calculate new dimensions
        new_w = int(w * zoom_factor)
        new_h = int(h * zoom_factor)
        
        # Resize with zoom using PyTorch bilinear interpolation
        img_zoomed = torch.nn.functional.interpolate(
            img_tensor.unsqueeze(0),
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Calculate crop position with slight drift
        if h > w:  # Vertical image - drift horizontally
            drift_x = int((new_w - video_w) * 0.15 * frame_progress)
            drift_y = 0
        else:  # Horizontal image - drift vertically
            drift_x = 0
            drift_y = int((new_h - video_h) * 0.15 * frame_progress)
        
        # Center crop with drift
        crop_x = max(0, min((new_w - video_w) // 2 + drift_x, new_w - video_w))
        crop_y = max(0, min((new_h - video_h) // 2 + drift_y, new_h - video_h))
        
        return img_zoomed[:, crop_y:crop_y+video_h, crop_x:crop_x+video_w]
    
    def _zoom_in(self, img_tensor: torch.Tensor, video_w: int, video_h: int,
                 frame_progress: float) -> torch.Tensor:
        """
        Pure zoom in effect
        
        Args:
            img_tensor: Image tensor (C, H, W)
            video_w: Target video width
            video_h: Target video height
            frame_progress: Progress through transition (0.0 to 1.0)
        
        Returns:
            Zoomed in tensor
        """
        c, h, w = img_tensor.shape
        
        zoom_factor = 1.0 + (self.intensity * frame_progress * 0.8)
        new_w = int(w * zoom_factor)
        new_h = int(h * zoom_factor)
        
        img_zoomed = torch.nn.functional.interpolate(
            img_tensor.unsqueeze(0),
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        crop_x = (new_w - video_w) // 2
        crop_y = (new_h - video_h) // 2
        
        return img_zoomed[:, crop_y:crop_y+video_h, crop_x:crop_x+video_w]
    
    def _zoom_out(self, img_tensor: torch.Tensor, video_w: int, video_h: int,
                  frame_progress: float) -> torch.Tensor:
        """
        Pure zoom out effect
        
        Args:
            img_tensor: Image tensor (C, H, W)
            video_w: Target video width
            video_h: Target video height
            frame_progress: Progress through transition (0.0 to 1.0)
        
        Returns:
            Zoomed out tensor
        """
        c, h, w = img_tensor.shape
        
        # Start zoomed in, zoom out to original
        zoom_factor = 1.0 + self.intensity * (1.0 - frame_progress) * 0.8
        new_w = int(w * zoom_factor)
        new_h = int(h * zoom_factor)
        
        img_zoomed = torch.nn.functional.interpolate(
            img_tensor.unsqueeze(0),
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        crop_x = (new_w - video_w) // 2
        crop_y = (new_h - video_h) // 2
        
        return img_zoomed[:, crop_y:crop_y+video_h, crop_x:crop_x+video_w]
    
    def _fade_in(self, img_tensor: torch.Tensor, video_w: int, video_h: int,
                 frame_progress: float) -> torch.Tensor:
        """
        Fade in effect - image gradually appears from black
        
        Args:
            img_tensor: Current image tensor
            video_w: Target width
            video_h: Target height
            frame_progress: 0.0 (black) to 1.0 (full image)
            
        Returns:
            Faded in image tensor
        """
        # Center crop if needed
        c, h, w = img_tensor.shape
        if h != video_h or w != video_w:
            crop_x = (w - video_w) // 2
            crop_y = (h - video_h) // 2
            img_tensor = img_tensor[:, crop_y:crop_y+video_h, crop_x:crop_x+video_w]
        
        # Fade from black (0) to full image (1)
        # Exponential fade for smoother transition
        fade_factor = frame_progress ** (1.0 + self.intensity * 2.0)
        return img_tensor * fade_factor
    
    def _fade_out(self, img_tensor: torch.Tensor, video_w: int, video_h: int,
                  frame_progress: float) -> torch.Tensor:
        """
        Fade out effect - image gradually disappears to black
        
        Args:
            img_tensor: Current image tensor
            video_w: Target width
            video_h: Target height
            frame_progress: 0.0 (full image) to 1.0 (black)
            
        Returns:
            Faded out image tensor
        """
        # Center crop if needed
        c, h, w = img_tensor.shape
        if h != video_h or w != video_w:
            crop_x = (w - video_w) // 2
            crop_y = (h - video_h) // 2
            img_tensor = img_tensor[:, crop_y:crop_y+video_h, crop_x:crop_x+video_w]
        
        # Fade from full image (1) to black (0)
        # Exponential fade for smoother transition
        fade_factor = (1.0 - frame_progress) ** (1.0 + self.intensity * 2.0)
        return img_tensor * fade_factor
    
    def _fade_cross(self, img_tensor: torch.Tensor, video_w: int, video_h: int,
                    frame_progress: float, prev_tensor: torch.Tensor) -> torch.Tensor:
        """
        Cross-fade effect - smooth transition between two images
        
        Args:
            img_tensor: Current image tensor
            video_w: Target width
            video_h: Target height
            frame_progress: 0.0 (previous image) to 1.0 (current image)
            prev_tensor: Previous image tensor
            
        Returns:
            Cross-faded image tensor
        """
        # Get device from input tensor
        device = img_tensor.device
        
        # Prepare current image (center crop if needed)
        c, h, w = img_tensor.shape
        if h != video_h or w != video_w:
            crop_x = (w - video_w) // 2
            crop_y = (h - video_h) // 2
            current_img = img_tensor[:, crop_y:crop_y+video_h, crop_x:crop_x+video_w]
        else:
            current_img = img_tensor
        
        # Prepare previous image (center crop if needed)
        # Ensure previous tensor is on same device
        prev_tensor = prev_tensor.to(device)
        c2, h2, w2 = prev_tensor.shape
        if h2 != video_h or w2 != video_w:
            crop_x2 = (w2 - video_w) // 2
            crop_y2 = (h2 - video_h) // 2
            prev_img = prev_tensor[:, crop_y2:crop_y2+video_h, crop_x2:crop_x2+video_w]
        else:
            prev_img = prev_tensor
        
        # Ensure both images have same dimensions
        if prev_img.shape != current_img.shape:
            prev_img = torch.nn.functional.interpolate(
                prev_img.unsqueeze(0),
                size=(video_h, video_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Smooth cross-fade with easing
        if frame_progress < 0.5:
            # First half: fade out previous image
            prev_weight = (1.0 - frame_progress * 2) ** (1.0 + self.intensity)
            current_weight = 0.0
        else:
            # Second half: fade in current image
            prev_weight = 0.0
            current_weight = ((frame_progress - 0.5) * 2) ** (1.0 + self.intensity)
        
        # If intensity is low, use linear cross-fade for simpler effect
        if self.intensity < 0.3:
            prev_weight = 1.0 - frame_progress
            current_weight = frame_progress
        
        return prev_img * prev_weight + current_img * current_weight
    
    def _dissolve(self, img_tensor: torch.Tensor, video_w: int, video_h: int,
                  frame_progress: float, prev_tensor: torch.Tensor) -> torch.Tensor:
        """
        Dissolve effect - pixelated transition between images
        
        Args:
            img_tensor: Current image tensor
            video_w: Target width
            video_h: Target height
            frame_progress: 0.0 (previous image) to 1.0 (current image)
            prev_tensor: Previous image tensor
            
        Returns:
            Dissolved image tensor
        """
        # Get device from input tensor
        device = img_tensor.device
        
        # Prepare current image (center crop if needed)
        c, h, w = img_tensor.shape
        if h != video_h or w != video_w:
            crop_x = (w - video_w) // 2
            crop_y = (h - video_h) // 2
            current_img = img_tensor[:, crop_y:crop_y+video_h, crop_x:crop_x+video_w]
        else:
            current_img = img_tensor
        
        # Prepare previous image (center crop if needed)
        c2, h2, w2 = prev_tensor.shape
        if h2 != video_h or w2 != video_w:
            crop_x2 = (w2 - video_w) // 2
            crop_y2 = (h2 - video_h) // 2
            prev_img = prev_tensor[:, crop_y2:crop_y2+video_h, crop_x2:crop_x2+video_w]
        else:
            prev_img = prev_tensor
        
        # Ensure both images have same dimensions
        if prev_img.shape != current_img.shape:
            prev_img = torch.nn.functional.interpolate(
                prev_img.unsqueeze(0),
                size=(video_h, video_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Create dissolve effect with noise pattern
        # Generate reproducible noise based on frame_progress
        # Set seed for reproducibility
        seed = int(frame_progress * 1000)
        torch.manual_seed(seed)
        
        # Create noise mask on the same device as input
        # Different grain sizes based on intensity
        if self.intensity > 0.7:
            # High intensity: fine grain noise (smooth dissolve)
            noise = torch.rand(1, video_h, video_w, device=device)
            block_size = 2
        elif self.intensity > 0.3:
            # Medium intensity: medium grain
            noise = torch.rand(1, video_h // 4, video_w // 4, device=device)
            noise = torch.nn.functional.interpolate(
                noise.unsqueeze(0),
                size=(video_h, video_w),
                mode='nearest'
            ).squeeze(0)
            block_size = 4
        else:
            # Low intensity: coarse grain (chunky dissolve)
            noise = torch.rand(1, video_h // 8, video_w // 8, device=device)
            noise = torch.nn.functional.interpolate(
                noise.unsqueeze(0),
                size=(video_h, video_w),
                mode='nearest'
            ).squeeze(0)
            block_size = 8
        
        # Threshold the noise based on progress
        # As progress increases, more pixels show current image
        threshold = frame_progress
        mask = (noise < threshold).float()
        
        # Expand mask to 3 channels (RGB)
        mask = mask.expand(3, video_h, video_w)
        
        # Apply dissolve: blend previous and current images based on mask
        result = prev_img * (1.0 - mask) + current_img * mask
        
        return result


# Legacy alias for backward compatibility
class TransitionEngine(TorchTransitionEngine):
    """Legacy alias"""
    pass