"""
Video processing module for Memories Creator
Handles video creation, encoding, and audio mixing
Now with enhanced progress logging and time tracking
"""

import os
import subprocess
import cv2
import numpy as np
import torch
import time
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from torchvision.transforms.functional import to_tensor

from .transitions import TorchTransitionEngine
from .media_utils import ExifParser, ImageProcessor 

class VideoProcessor:
    """Handles video creation and processing"""
    
    def __init__(self, config, device: torch.device):
        """
        Initialize video processor
        
        Args:
            config: Configuration object
            device: Torch device (mps, cuda, or cpu)
        """
        self.config = config
        self.device = device
        
        # Choose resize algorithm
        self.resize_algo = self.config.get('resize_algorithm', 'bicubic')
        if self.resize_algo == 'bilinear':
            self.pil_resample = Image.Resampling.BILINEAR
        elif self.resize_algo == 'bicubic':
            self.pil_resample = Image.Resampling.BICUBIC
        else:  # lanczos
            self.pil_resample = Image.Resampling.LANCZOS
        
        # Timing tracking
        self.step_start_time = None
    
    def _log_step_start(self, message: str):
        """Log start of processing step with timestamp"""
        self.step_start_time = time.time()
        timestamp = time.strftime("%H:%M:%S")
        print(f"\n[{timestamp}] {message}")
    
    def _log_step_end(self, message: str = "Complete"):
        """Log end of processing step with elapsed time"""
        if self.step_start_time:
            elapsed = time.time() - self.step_start_time
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {message} (took {elapsed:.1f}s)")
            self.step_start_time = None
    
    def _log_info(self, message: str):
        """Log informational message with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def create_video(self, images_info: List[Dict], video_w: int, video_h: int,
                    output_path: str) -> bool:
        """
        Create video from images
        
        Args:
            images_info: List of image metadata dictionaries
            video_w: Video width
            video_h: Video height
            output_path: Path for output video
            
        Returns:
            True if successful
        """
        transition_type = self.config.get('transition_type', 'static')
        
        if transition_type == 'static':
            # FAST PATH: static video - no movement
            return self._create_static_video(images_info, video_w, video_h, output_path)
        elif transition_type == 'pan':
            # FAST PATH: pan video - smooth camera movement with gentle zoom
            return self._create_enhanced_pan_video(images_info, video_w, video_h, output_path)
        else:
            # STANDARD PATH: complex transitions with PyTorch
            return self._create_transition_video(images_info, video_w, video_h, output_path)
    
    def _get_video_writer(self, output_path: str, fps: int, video_w: int, video_h: int):
        """
        Initialize video writer with optimal codec
        
        Args:
            output_path: Output file path
            fps: Frames per second
            video_w: Video width
            video_h: Video height
            
        Returns:
            Initialized VideoWriter or None
        """
        codec_preference = self.config.get('video_codec', 'libx264')
        
        # Try preferred codec first
        if codec_preference in ['libx264', 'h264_videotoolbox', 'hevc_videotoolbox']:
            # Use ffmpeg pipeline for VideoToolbox acceleration
            return None  # Signal to use ffmpeg pipeline
        
        # Try hardware codec (hvc1 or avc1)
        fourcc = cv2.VideoWriter_fourcc(*codec_preference)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (video_w, video_h))
        
        if writer.isOpened():
            return writer
        
        # Fallback to avc1
        print(f"⚠️  Could not use {codec_preference}, falling back to avc1")
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (video_w, video_h))
        
        if writer.isOpened():
            return writer
        
        print("❌ Failed to initialize video writer")
        return None
    
    def _create_static_video(self, images_info: List[Dict], video_w: int, video_h: int,
                            output_path: str) -> bool:
        """
        Create static video - each photo is centered and fixed
        Fastest mode, no movement
        """
        self._log_step_start(f"🎬 Creating STATIC video ({video_w}x{video_h} @ {self.config.get('fps')} FPS)")
        self._log_info("   Mode: Static (no movement, fastest)")
        
        fps = self.config.get('fps', 30)
        codec = self.config.get('video_codec', 'libx264')
        
        # Use ffmpeg pipeline for better performance
        result = self._create_static_video_ffmpeg(images_info, video_w, video_h, output_path)
        
        if result:
            self._log_step_end("✅ Static video creation complete")
        
        return result
    
    def _create_static_video_ffmpeg(self, images_info: List[Dict], video_w: int, video_h: int,
                                    output_path: str) -> bool:
        """
        Create static video using ffmpeg with optimized codec
        """
        fps = self.config.get('fps', 30)
        seconds_per_photo = self.config.get('seconds_per_photo', 3)
        frames_per_photo = int(seconds_per_photo * fps)
        
        codec = self.config.get('video_codec', 'libx264')
        self._log_info(f"   Codec: {codec}")
        self._log_info(f"   Duration per photo: {seconds_per_photo}s ({frames_per_photo} frames)")
        
        # Create temporary directory for frames
        temp_dir = os.path.join(os.path.dirname(output_path), 'temp_frames')
        os.makedirs(temp_dir, exist_ok=True)
        
        self._log_step_start("📥 Preparing frames")
        
        total_frames = len(images_info) * frames_per_photo
        pbar = tqdm(total=len(images_info), 
                   desc="Processing images", 
                   ncols=100, 
                   unit="img",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        frame_count = 0
        for idx, img_info in enumerate(images_info):
            try:
                frame_bgr = self._load_and_scale_image(img_info, video_w, video_h)
                
                # Write this frame multiple times
                for _ in range(frames_per_photo):
                    frame_path = os.path.join(temp_dir, f'frame_{frame_count:06d}.jpg')
                    # Use JPEG for faster I/O
                    cv2.imwrite(frame_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    frame_count += 1
                
            except Exception as e:
                self._log_info(f"\n⚠️  Warning: Could not load slide {idx}: {e}")
            
            pbar.update(1)
        
        pbar.close()
        self._log_step_end(f"Prepared {frame_count} frames")
        
        # Encode with ffmpeg using optimized codec
        self._log_step_start("📹 Encoding video with ffmpeg")
        
        # Prepare ffmpeg command based on codec
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%06d.jpg'),
        ]
        
        # Add codec-specific parameters
        if codec == 'h264_videotoolbox':
            cmd.extend([
                '-c:v', 'h264_videotoolbox',
                '-b:v', '10M',  # Bitrate for quality
                '-profile:v', 'high',
            ])
            self._log_info("   Using H.264 VideoToolbox (hardware acceleration)")
        elif codec == 'hevc_videotoolbox':
            cmd.extend([
                '-c:v', 'hevc_videotoolbox',
                '-b:v', '8M',  # HEVC is more efficient
                '-tag:v', 'hvc1',
            ])
            self._log_info("   Using HEVC VideoToolbox (hardware acceleration)")
        else:  # libx264 fallback
            cmd.extend([
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', str(self.config.get('video_quality', 23)),
            ])
            self._log_info("   Using libx264 (CPU encoding)")
        
        # Common parameters
        cmd.extend([
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            output_path
        ])
        
        try:
            # Run with progress bar for ffmpeg
            process = subprocess.Popen(
                cmd,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Monitor ffmpeg progress
            for line in process.stderr:
                if 'frame=' in line:
                    # Extract frame number for progress tracking
                    parts = line.split('frame=')
                    if len(parts) > 1:
                        try:
                            current_frame = int(parts[1].split()[0])
                            progress_pct = (current_frame / total_frames) * 100
                            print(f"\r   Encoding: {progress_pct:.1f}% ({current_frame}/{total_frames} frames)", end='')
                        except:
                            pass
            
            process.wait()
            
            if process.returncode == 0:
                print()  # New line after progress
                self._log_step_end("Video encoding complete")
                
                # Cleanup temp frames
                import shutil
                shutil.rmtree(temp_dir)
                
                return True
            else:
                print(f"\n❌ Encoding failed with code {process.returncode}")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Encoding failed: {e}")
            # Try fallback to libx264
            if codec in ['h264_videotoolbox', 'hevc_videotoolbox']:
                self._log_info("⚠️  VideoToolbox failed, falling back to libx264")
                self.config.set('video_codec', 'libx264')
                return self._create_static_video_ffmpeg(images_info, video_w, video_h, output_path)
            return False
    
    def _load_and_scale_image(self, img_info: Dict, video_w: int, video_h: int) -> np.ndarray:
        """
        Load image and scale to video dimensions with all rotations applied
        
        Args:
            img_info: Image metadata dictionary
            video_w: Target video width
            video_h: Target video height
            
        Returns:
            BGR numpy array ready for OpenCV
        """
        # Load with all rotations and enhancements applied
        img = ImageProcessor.load_and_prepare_image(
            filepath=img_info['path'],
            custom_rotation=img_info.get('custom_rotation', 0),
            enhance_old=self.config.get('enhance_old_photos', False),
            is_old=img_info.get('is_old', False),
            enhance_strength=self.config.get('old_photo_enhancement_strength', 0.03)
        )
        
        if img is None:
            # Return black frame if image fails to load
            return np.zeros((video_h, video_w, 3), dtype=np.uint8)
        
        # Scale and crop to exact video dimensions
        img = ImageProcessor.resize_and_crop(img, video_w, video_h, self.resize_algo)
        
        # Convert to BGR for OpenCV
        img_np = np.array(img)
        frame_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        return frame_bgr
    
    def _create_enhanced_pan_video(self, images_info: List[Dict], video_w: int, video_h: int,
                                  output_path: str) -> bool:
        """
        Create pan video with smooth camera movement and gentle zoom
        Fast mode using OpenCV (no PyTorch tensors)
        """
        self._log_step_start(f"🎬 Creating PAN video ({video_w}x{video_h} @ {self.config.get('fps')} FPS)")
        self._log_info("   Mode: Enhanced Pan (movement + gentle zoom)")
        
        fps = self.config.get('fps', 30)
        seconds_per_photo = self.config.get('seconds_per_photo', 3)
        frames_per_photo = int(seconds_per_photo * fps)
        codec = self.config.get('video_codec', 'libx264')
        
        self._log_info(f"   Codec: {codec}")
        self._log_info(f"   Duration per photo: {seconds_per_photo}s ({frames_per_photo} frames)")
        
        # Determine if we need ffmpeg pipeline
        use_ffmpeg = codec in ['libx264', 'h264_videotoolbox', 'hevc_videotoolbox']
        
        if use_ffmpeg:
            temp_dir = os.path.join(os.path.dirname(output_path), 'temp_frames')
            os.makedirs(temp_dir, exist_ok=True)
        else:
            writer = self._get_video_writer(output_path, fps, video_w, video_h)
            if writer is None:
                return False
        
        # Process each image
        self._log_step_start("📥 Generating frames with pan effect")
        
        total_frames = len(images_info) * frames_per_photo
        pbar = tqdm(total=len(images_info),
                   desc="Processing images",
                   ncols=100,
                   unit="img",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        frame_count = 0
        pan_intensity = self.config.get('transition_intensity', 0.1) * 0.5
        
        for idx, img_info in enumerate(images_info):
            try:
                # Load with rotations
                img = ImageProcessor.load_and_prepare_image(
                    filepath=img_info['path'],
                    custom_rotation=img_info.get('custom_rotation', 0),
                    enhance_old=self.config.get('enhance_old_photos', False),
                    is_old=img_info.get('is_old', False),
                    enhance_strength=self.config.get('old_photo_enhancement_strength', 0.03)
                )
                
                if img is None:
                    continue
                
                # Scale with margin for pan effect
                w, h = img.size
                scale = max(video_w / w, video_h / h)
                margin = 1.0 + pan_intensity
                new_w = int(w * scale * margin)
                new_h = int(h * scale * margin)
                
                img_scaled = img.resize((new_w, new_h), self.pil_resample)
                img_np = np.array(img_scaled)
                
                # Generate frames with pan effect
                orientation = img_info.get('orientation', 'horizontal')
                
                for i in range(frames_per_photo):
                    progress = i / max(frames_per_photo - 1, 1)
                    
                    # Calculate pan offset based on orientation
                    if orientation == 'vertical':
                        # Pan left to right
                        max_offset_x = new_w - video_w
                        offset_x = int(max_offset_x * progress)
                        offset_y = (new_h - video_h) // 2
                    else:
                        # Pan top to bottom
                        offset_x = (new_w - video_w) // 2
                        max_offset_y = new_h - video_h
                        offset_y = int(max_offset_y * progress)
                    
                    # Crop frame
                    frame_rgb = img_np[offset_y:offset_y+video_h, offset_x:offset_x+video_w]
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    
                    # Write frame
                    if use_ffmpeg:
                        frame_path = os.path.join(temp_dir, f'frame_{frame_count:06d}.jpg')
                        cv2.imwrite(frame_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    else:
                        writer.write(frame_bgr)
                    
                    frame_count += 1
                
            except Exception as e:
                self._log_info(f"\n⚠️  Warning: Could not process slide {idx}: {e}")
            
            pbar.update(1)
        
        pbar.close()
        self._log_step_end(f"Generated {frame_count} frames")
        
        # Finalize
        if use_ffmpeg:
            self._log_step_start("📹 Encoding video with ffmpeg")
            
            # Prepare ffmpeg command
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%06d.jpg'),
            ]
            
            # Add codec-specific parameters
            if codec == 'h264_videotoolbox':
                cmd.extend([
                    '-c:v', 'h264_videotoolbox',
                    '-b:v', '10M',
                    '-profile:v', 'high',
                ])
                self._log_info("   Using H.264 VideoToolbox")
            elif codec == 'hevc_videotoolbox':
                cmd.extend([
                    '-c:v', 'hevc_videotoolbox',
                    '-b:v', '8M',
                    '-tag:v', 'hvc1',
                ])
                self._log_info("   Using HEVC VideoToolbox")
            else:  # libx264
                cmd.extend([
                    '-c:v', 'libx264',
                    '-preset', 'medium',
                    '-crf', str(self.config.get('video_quality', 23)),
                ])
                self._log_info("   Using libx264")
            
            # Common parameters
            cmd.extend([
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                output_path
            ])
            
            try:
                process = subprocess.Popen(
                    cmd,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                # Monitor progress
                for line in process.stderr:
                    if 'frame=' in line:
                        parts = line.split('frame=')
                        if len(parts) > 1:
                            try:
                                current_frame = int(parts[1].split()[0])
                                progress_pct = (current_frame / total_frames) * 100
                                print(f"\r   Encoding: {progress_pct:.1f}% ({current_frame}/{total_frames} frames)", end='')
                            except:
                                pass
                
                process.wait()
                print()  # New line
                
                if process.returncode == 0:
                    # Cleanup
                    import shutil
                    shutil.rmtree(temp_dir)
                    
                    self._log_step_end("Video encoding complete")
                    return True
                else:
                    return False
                    
            except subprocess.CalledProcessError as e:
                print(f"\n❌ Encoding failed: {e}")
                # Try fallback
                if codec in ['h264_videotoolbox', 'hevc_videotoolbox']:
                    self._log_info("⚠️  VideoToolbox failed, falling back to libx264")
                    self.config.set('video_codec', 'libx264')
                    return self._create_enhanced_pan_video(images_info, video_w, video_h, output_path)
                return False
        else:
            writer.release()
            self._log_step_end("Video creation complete")
            return True
    
    def _create_transition_video(self, images_info: List[Dict], video_w: int, video_h: int,
                                output_path: str) -> bool:
        """
        Create video with complex GPU-accelerated transitions
        Uses PyTorch tensors for effects like Ken Burns, zoom, fade
        """
        self._log_step_start(f"🎬 Creating TRANSITION video ({video_w}x{video_h} @ {self.config.get('fps')} FPS)")
        
        fps = self.config.get('fps', 30)
        seconds_per_photo = self.config.get('seconds_per_photo', 3)
        frames_per_photo = int(seconds_per_photo * fps)
        transition_type = self.config.get('transition_type', 'ken_burns')
        transition_intensity = self.config.get('transition_intensity', 0.1)
        codec = self.config.get('video_codec', 'libx264')
        
        self._log_info(f"   Codec: {codec}")
        self._log_info(f"   Transition: {transition_type} (intensity: {transition_intensity})")
        if transition_type in ['fade_cross', 'dissolve']:
            self._log_info(f"   Note: First slide fades in from black")
        self._log_info(f"   Duration per photo: {seconds_per_photo}s ({frames_per_photo} frames)")
        self._log_info(f"   Device: {self.device}")
        
        # Determine if we need ffmpeg pipeline
        use_ffmpeg = codec in ['libx264', 'h264_videotoolbox', 'hevc_videotoolbox']
        
        if use_ffmpeg:
            temp_dir = os.path.join(os.path.dirname(output_path), 'temp_frames')
            os.makedirs(temp_dir, exist_ok=True)
        else:
            writer = self._get_video_writer(output_path, fps, video_w, video_h)
            if writer is None:
                return False
        
        # Pre-load all images as tensors
        self._log_step_start("📥 Loading images to GPU memory")
        
        image_tensors = []
        for idx, img_info in enumerate(tqdm(images_info, 
                                            desc="Loading", 
                                            ncols=80, 
                                            unit="img")):
            tensor = self._load_and_prepare_image(img_info, video_w, video_h)
            if tensor is not None:
                image_tensors.append(tensor)
            else:
                self._log_info(f"\n⚠️  Warning: Could not load slide {idx}")
        
        self._log_step_end(f"Loaded {len(image_tensors)} images to {self.device}")
        
        if not image_tensors:
            print("❌ No valid images loaded")
            return False
        
        # Generate frames with transitions
        self._log_step_start(f"🎨 Generating frames with {transition_type} transition")
        
        total_frames = len(image_tensors) * frames_per_photo
        pbar = tqdm(total=len(image_tensors),
                   desc="Generating",
                   ncols=100,
                   unit="img",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        frame_count = 0
        success = True
        
        # Check if transition needs previous image
        needs_prev = transition_type in ['fade_cross', 'dissolve']
        total_images = len(image_tensors)
        
        for img_idx, img_tensor in enumerate(image_tensors):
            try:
                # Determine transition type for this slide
                prev_tensor = None
                current_transition = transition_type
                
                if needs_prev:
                    if img_idx == 0:
                        # First slide: fade in from black (no previous image)
                        current_transition = 'fade_in'
                        prev_tensor = None
                    else:
                        # All other slides: cross-fade from previous
                        current_transition = transition_type
                        prev_tensor = image_tensors[img_idx - 1]
                
                # Create transition engine for this image
                transition_engine = TorchTransitionEngine(
                    transition_type=current_transition,
                    intensity=transition_intensity,
                    prev_image=prev_tensor
                )
                
                # Generate frames for this slide
                for i in range(frames_per_photo):
                    progress = i / max(frames_per_photo - 1, 1)
                    
                    # Apply transition
                    frame_tensor = transition_engine.apply(
                        img_tensor,
                        video_w,
                        video_h,
                        progress,
                        prev_tensor=prev_tensor
                    )
                    
                    # Convert to OpenCV format
                    frame_bgr = self._tensor_to_frame(frame_tensor)
                    
                    # Apply vignette if requested
                    vignette_strength = self.config.get('vignette_strength', 0.0)
                    if vignette_strength > 0:
                        frame_bgr = self._apply_vignette(frame_bgr, vignette_strength)
                    
                    # Write frame
                    if use_ffmpeg:
                        frame_path = os.path.join(temp_dir, f'frame_{frame_count:06d}.jpg')
                        cv2.imwrite(frame_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    else:
                        writer.write(frame_bgr)
                    
                    frame_count += 1
                
            except Exception as e:
                self._log_info(f"\n⚠️  Warning: Error processing slide {img_idx}: {e}")
                success = False
                break
            
            pbar.update(1)
        
        pbar.close()
        self._log_step_end(f"Generated {frame_count} frames")
        
        # Finalize
        if use_ffmpeg and success:
            self._log_step_start("📹 Encoding video with ffmpeg")
            
            # Prepare ffmpeg command
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%06d.jpg'),
            ]
            
            # Add codec-specific parameters
            if codec == 'h264_videotoolbox':
                cmd.extend([
                    '-c:v', 'h264_videotoolbox',
                    '-b:v', '10M',
                    '-profile:v', 'high',
                ])
                self._log_info("   Using H.264 VideoToolbox")
            elif codec == 'hevc_videotoolbox':
                cmd.extend([
                    '-c:v', 'hevc_videotoolbox',
                    '-b:v', '8M',
                    '-tag:v', 'hvc1',
                ])
                self._log_info("   Using HEVC VideoToolbox")
            else:  # libx264
                cmd.extend([
                    '-c:v', 'libx264',
                    '-preset', 'medium',
                    '-crf', str(self.config.get('video_quality', 23)),
                ])
                self._log_info("   Using libx264")
            
            # Common parameters
            cmd.extend([
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                output_path
            ])
            
            try:
                process = subprocess.Popen(
                    cmd,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                # Monitor progress
                for line in process.stderr:
                    if 'frame=' in line:
                        parts = line.split('frame=')
                        if len(parts) > 1:
                            try:
                                current_frame = int(parts[1].split()[0])
                                progress_pct = (current_frame / total_frames) * 100
                                print(f"\r   Encoding: {progress_pct:.1f}% ({current_frame}/{total_frames} frames)", end='')
                            except:
                                pass
                
                process.wait()
                print()  # New line
                
                if process.returncode == 0:
                    # Cleanup
                    import shutil
                    shutil.rmtree(temp_dir)
                    
                    success = True
                else:
                    success = False
                    
            except subprocess.CalledProcessError as e:
                print(f"\n❌ Encoding failed: {e}")
                # Try fallback
                if codec in ['h264_videotoolbox', 'hevc_videotoolbox']:
                    self._log_info("⚠️  VideoToolbox failed, falling back to libx264")
                    original_codec = self.config.get('video_codec')
                    self.config.set('video_codec', 'libx264')
                    success = self._create_transition_video(images_info, video_w, video_h, output_path)
                    self.config.set('video_codec', original_codec)
                else:
                    success = False
        else:
            if not use_ffmpeg:
                writer.release()
        
        # Clean up GPU memory
        if self.device.type == 'mps':
            torch.mps.empty_cache()
        elif self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        if success:
            self._log_step_end("Video creation complete")
        
        return success

    def _load_and_prepare_image(self, img_info: Dict, video_w: int, 
                                video_h: int) -> Optional[torch.Tensor]:
        """
        Load image and prepare tensor for transitions with ALL rotations applied
        """
        try:
            # Use ImageProcessor to load with all rotations
            img = ImageProcessor.load_and_prepare_image(
                filepath=img_info['path'],
                custom_rotation=img_info.get('custom_rotation', 0),
                enhance_old=self.config.get('enhance_old_photos', False),
                is_old=img_info.get('is_old', False),
                enhance_strength=self.config.get('old_photo_enhancement_strength', 0.03)
            )
            
            if img is None:
                return None
            
            # Scale with margin for transitions
            w, h = img.size
            scale = max(video_w / w, video_h / h)
            
            # Add margin for transition effects
            transition_scale = 1.0 + self.config.get('transition_intensity', 0.1)
            new_w = int(w * scale * transition_scale)
            new_h = int(h * scale * transition_scale)
            
            img = img.resize((new_w, new_h), self.pil_resample)
            
            # Convert to tensor
            img_tensor = to_tensor(img).to(self.device)
            
            return img_tensor
            
        except Exception as e:
            print(f"\n⚠️  Warning: Could not load {img_info['path']}: {e}")
            return None

    def _apply_vignette(self, frame_bgr: np.ndarray, strength: float) -> np.ndarray:
        """
        Apply subtle vignette effect to frame
        
        Args:
            frame_bgr: Input BGR frame
            strength: Vignette strength (0.0 to 1.0)
            
        Returns:
            Frame with vignette applied
        """
        h, w = frame_bgr.shape[:2]
        
        # Create radial gradient mask
        center_x, center_y = w // 2, h // 2
        y, x = np.ogrid[:h, :w]
        
        # Calculate distance from center (normalized)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2) / max_dist
        
        # Create vignette mask (1.0 at center, strength at edges)
        vignette = 1.0 - (dist * strength)
        vignette = np.clip(vignette, 0.0, 1.0)
        
        # Apply vignette
        frame_float = frame_bgr.astype(np.float32)
        frame_float[:, :, 0] *= vignette
        frame_float[:, :, 1] *= vignette
        frame_float[:, :, 2] *= vignette
        
        return np.clip(frame_float, 0, 255).astype(np.uint8)
    
    def _tensor_to_frame(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to OpenCV BGR frame"""
        frame_np = tensor.cpu().permute(1, 2, 0).numpy()
        frame_np = (frame_np * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        return frame_bgr
    
    def add_audio(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Add audio track to video"""
        if not os.path.exists(audio_path):
            print(f"⚠️  Audio file not found: {audio_path}")
            return False
        
        self._log_step_start("🎵 Adding audio track")
        
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', audio_path,
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-c:v', 'copy',
            '-c:a', self.config.get('audio_codec', 'aac'),
            '-b:a', self.config.get('audio_bitrate', '192k'),
            '-shortest',
            '-movflags', '+faststart',
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            self._log_step_end("Audio track added successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error adding audio: {e.stderr}")
            return False
    
    def replace_audio(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Replace audio track in existing video"""
        self._log_step_start("🔄 Replacing audio track")
        
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', audio_path,
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-c:v', 'copy',
            '-c:a', self.config.get('audio_codec', 'aac'),
            '-b:a', self.config.get('audio_bitrate', '192k'),
            '-shortest',
            '-movflags', '+faststart',
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            self._log_step_end("Audio replaced successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error replacing audio: {e.stderr}")
            return False
