"""
Advanced features for Memories Creator
Handles slide rotation, reordering, and video reassembly
Now with intelligent handling of cross-fade transitions
"""

import os
import subprocess
import json
import time
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


class SlideEditor:
    """Handles editing individual slides in existing videos"""
    
    # Transitions that bake previous frame into current frame
    CROSS_FADE_TRANSITIONS = ['fade_cross', 'dissolve']
    
    def __init__(self, config, video_path: str, config_path: str):
        """
        Initialize slide editor
        
        Args:
            config: Configuration object
            video_path: Path to existing video
            config_path: Path to video configuration JSON
        """
        self.config = config
        self.video_path = video_path
        self.config_path = config_path
        self.temp_dir = os.path.join(
            config.expand_path('output_dir'),
            'temp_slides'
        )
        
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Logging helpers
        self.start_time = None
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
    
    def load_video_config(self) -> Optional[Dict]:
        """Load video configuration"""
        if not os.path.exists(self.config_path):
            print(f"❌ Configuration not found: {self.config_path}")
            return None
        
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return None
    
    def requires_full_rebuild(self, rotate_config: Dict[int, int], 
                             video_config: Dict) -> bool:
        """
        Determine if video requires full rebuild or can use segment rotation
        
        Cross-fade transitions (fade_cross, dissolve) bake the previous frame
        into the current slide. When rotating slides with these transitions,
        we must rebuild the entire video to ensure transitions are regenerated
        with correctly rotated images.
        
        Args:
            rotate_config: Dict mapping slide indices to rotation degrees
            video_config: Video configuration with transition info
            
        Returns:
            True if full rebuild required, False if segment rotation is OK
        """
        transition_type = video_config.get('transition_type', 'ken_burns')
        
        # If using cross-fade transition, always rebuild
        if transition_type in self.CROSS_FADE_TRANSITIONS:
            return True
        
        # For other transitions, segment rotation is safe
        return False
    
    def rebuild_video_with_rotations(self, rotate_config: Dict[int, int],
                                    video_config: Dict) -> bool:
        """
        Rebuild entire video from scratch with rotations applied
        
        This is required for cross-fade transitions where the previous frame
        is baked into the current slide during transition generation.
        
        Args:
            rotate_config: Dict mapping slide indices to rotation degrees
            video_config: Video configuration
            
        Returns:
            True if successful
        """
        self._log_step_start("🔄 Full video rebuild required (cross-fade transition detected)")
        
        # Update video config with rotations
        for slide_idx, degrees in rotate_config.items():
            if slide_idx < len(video_config['images']):
                video_config['images'][slide_idx]['custom_rotation'] = degrees
                self._log_info(f"   Slide {slide_idx}: {degrees}° rotation")
        
        # Save updated config
        with open(self.config_path, 'w') as f:
            json.dump(video_config, f, indent=2)
        
        self._log_step_end("Configuration updated")
        
        # Import here to avoid circular dependency
        from video_processor import VideoProcessor
        from media_utils import ImageAnalyzer
        
        self._log_step_start("📸 Reloading and preparing images with rotations")
        
        # Prepare images_info list from config
        images_info = []
        year_threshold = self.config.get('year_threshold', 2020)
        
        for img_config in tqdm(video_config['images'], 
                              desc="Analyzing", 
                              ncols=80, 
                              unit="img"):
            filepath = img_config['path']
            
            # Analyze image (gets EXIF, dimensions, etc.)
            info = ImageAnalyzer.analyze_image(filepath, year_threshold)
            
            if info:
                # Apply custom rotation from updated config
                info['custom_rotation'] = img_config.get('custom_rotation', 0)
                images_info.append(info)
            else:
                print(f"⚠️  Warning: Could not analyze {filepath}")
        
        self._log_step_end(f"Prepared {len(images_info)} images")
        
        # Get video dimensions from config
        video_w = video_config['video_width']
        video_h = video_config['video_height']
        
        # Initialize video processor
        import torch
        use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
        device = torch.device('mps' if use_mps else 'cpu')
        
        video_processor = VideoProcessor(self.config, device)
        
        # Create temporary video (without audio)
        temp_output = self.video_path.replace('.mp4', '_rebuilding.mp4')
        
        self._log_step_start(f"🎬 Generating video with corrected rotations")
        self._log_info(f"   Resolution: {video_w}x{video_h}")
        self._log_info(f"   Transition: {video_config.get('transition_type', 'unknown')}")
        
        success = video_processor.create_video(
            images_info,
            video_w,
            video_h,
            temp_output
        )
        
        if not success:
            print("❌ Video generation failed")
            return False
        
        self._log_step_end("Video generation complete")
        
        # Restore audio track
        final_output = self.video_path.replace('.mp4', '_temp_final.mp4')
        audio_restored = False
        
        # Try config music first
        music_file = self.config.expand_path('music_file')
        if music_file and os.path.exists(music_file):
            self._log_step_start(f"🎵 Adding audio: {os.path.basename(music_file)}")
            if self._add_audio_track(temp_output, music_file, final_output):
                audio_restored = True
                self._log_step_end("Audio added")
        
        # Fallback to original video audio
        if not audio_restored and os.path.exists(self.video_path):
            self._log_step_start("🎵 Extracting and restoring original audio")
            temp_audio = os.path.join(self.temp_dir, 'original_audio.aac')
            
            extract_cmd = [
                'ffmpeg', '-y',
                '-i', self.video_path,
                '-vn',
                '-acodec', 'copy',
                temp_audio
            ]
            
            try:
                subprocess.run(extract_cmd, check=True, 
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
                
                if self._add_audio_track(temp_output, temp_audio, final_output):
                    audio_restored = True
                    self._log_step_end("Audio restored")
                
                try:
                    os.remove(temp_audio)
                except:
                    pass
                    
            except subprocess.CalledProcessError:
                self._log_info("   ⚠️  No audio found in original video")
        
        # Replace original video
        os.remove(self.video_path)
        
        if audio_restored and os.path.exists(final_output):
            os.rename(final_output, self.video_path)
            try:
                os.remove(temp_output)
            except:
                pass
        else:
            os.rename(temp_output, self.video_path)
            self._log_info("   ℹ️  Video created without audio")
        
        return True
    
    def get_slide_timecodes(self, video_config: Dict) -> List[float]:
        """
        Calculate start time for each slide
        
        Returns:
            List of start times in seconds
        """
        timecodes = []
        current_time = 0.0
        
        for img_config in video_config['images']:
            timecodes.append(current_time)
            current_time += img_config['duration_seconds']
        
        return timecodes
    
    def extract_slide(self, slide_index: int, video_config: Dict,
                     output_path: str) -> bool:
        """
        Extract a single slide from video
        
        Args:
            slide_index: Index of slide to extract
            video_config: Video configuration dict
            output_path: Where to save extracted slide
            
        Returns:
            True if successful
        """
        timecodes = self.get_slide_timecodes(video_config)
        
        if slide_index >= len(timecodes):
            print(f"❌ Slide {slide_index} does not exist")
            return False
        
        start_time = timecodes[slide_index]
        duration = video_config['images'][slide_index]['duration_seconds']
        
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_time),
            '-i', self.video_path,
            '-t', str(duration),
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '18',
            '-an',  # No audio for slide segments
            output_path
        ]
        
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Extraction failed: {e}")
            return False
    
    def rotate_slide(self, input_path: str, output_path: str,
                    degrees: int) -> bool:
        """
        Rotate a video slide
        
        Args:
            input_path: Input video segment
            output_path: Output rotated video
            degrees: Rotation angle (90, 180, 270)
            
        Returns:
            True if successful
        """
        if degrees not in [90, 180, 270]:
            print(f"❌ Invalid rotation: {degrees}° (use 90, 180, or 270)")
            return False
        
        if degrees == 0:
            # No rotation needed
            cmd = ['cp', input_path, output_path]
        elif degrees == 90:
            transpose = 'transpose=1'
        elif degrees == 180:
            transpose = 'transpose=1,transpose=1'
        else:  # 270
            transpose = 'transpose=2'
        
        if degrees == 0:
            subprocess.run(cmd)
            return True
        
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-vf', transpose,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '18',
            '-an',
            output_path
        ]
        
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Rotation failed: {e}")
            return False
    
    def assemble_video(self, slide_paths: List[str], output_path: str) -> bool:
        """
        Assemble slides into final video
        
        Args:
            slide_paths: List of slide video paths in order
            output_path: Output video path
            
        Returns:
            True if successful
        """
        # Create concat file
        concat_file = os.path.join(self.temp_dir, 'concat_list.txt')
        
        with open(concat_file, 'w') as f:
            for slide_path in slide_paths:
                abs_path = os.path.abspath(slide_path)
                f.write(f"file '{abs_path}'\n")
        
        # Concatenate slides
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',
            '-movflags', '+faststart',
            output_path
        ]
        
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Cleanup
            os.remove(concat_file)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Assembly failed: {e}")
            return False
    
    def apply_config_edits(self) -> bool:
        """
        Apply slide edits from config.json (arrange and rotate fields)
        
        Returns:
            True if successful
        """
        self.start_time = time.time()
        
        self._log_step_start("⚙️  Checking config.json for slide edits")
        
        # Get arrange and rotate from config
        arrange_spec = self.config.get('arrange', '').strip()
        rotate_spec = self.config.get('rotate', '').strip()
        
        if not arrange_spec and not rotate_spec:
            self._log_info("   ℹ️  No edits specified (arrange and rotate are empty)")
            return False
        
        print("\n" + "=" * 70)
        print("⚡ Applying Config-Based Slide Edits")
        print("=" * 70)
        
        if arrange_spec:
            self._log_info(f"📋 Arrange: {arrange_spec}")
        if rotate_spec:
            self._log_info(f"🔄 Rotate: {rotate_spec}")
        
        # Load video config
        video_config = self.load_video_config()
        if not video_config:
            return False
        
        total_slides = len(video_config['images'])
        transition_type = video_config.get('transition_type', 'unknown')
        
        self._log_info(f"📊 Total slides: {total_slides}")
        self._log_info(f"🎨 Transition type: {transition_type}")
        
        # Parse arrange specification
        new_order = None
        if arrange_spec:
            try:
                new_order = self._parse_arrange_spec(arrange_spec, total_slides)
                self._log_info(f"✅ Parsed arrange: {len(new_order)} slides")
            except ValueError as e:
                print(f"❌ Error parsing arrange: {e}")
                return False
        
        # Parse rotate specification
        rotate_config = {}
        if rotate_spec:
            try:
                rotate_config = parse_rotate_spec(rotate_spec)
                self._log_info(f"✅ Parsed rotate: {len(rotate_config)} operations")
                
                # Validate slide indices
                invalid_slides = [idx for idx in rotate_config.keys() if idx >= total_slides]
                if invalid_slides:
                    print(f"❌ Invalid slide indices: {invalid_slides}")
                    print(f"   Video has {total_slides} slides (0-{total_slides-1})")
                    return False
                    
            except ValueError as e:
                print(f"❌ Error parsing rotate: {e}")
                return False
        
        # Apply reordering first if specified
        if new_order:
            self._log_step_start("📋 Reordering slides")
            
            # Reorder images in config
            original_images = video_config['images']
            video_config['images'] = [original_images[i] for i in new_order]
            
            # Update indices
            for idx, img_config in enumerate(video_config['images']):
                img_config['index'] = idx
            
            # Save updated config
            with open(self.config_path, 'w') as f:
                json.dump(video_config, f, indent=2)
            
            self._log_step_end(f"Slides reordered: {arrange_spec}")
            
            # Note: Actual video reassembly would happen here
            # For now, just update config
            self._log_info("⚠️  Note: Arrange requires manual video rebuild")
        
        # Apply rotations
        if rotate_config:
            # Check if full rebuild is required
            needs_rebuild = self.requires_full_rebuild(rotate_config, video_config)
            
            if needs_rebuild:
                self._log_info(f"🔄 Cross-fade transition detected: '{transition_type}'")
                self._log_info("   Full video rebuild required to regenerate transitions")
                
                success = self.rebuild_video_with_rotations(rotate_config, video_config)
                
                if success:
                    total_elapsed = time.time() - self.start_time
                    print("\n" + "=" * 70)
                    print(f"✅ Video rebuild complete in {total_elapsed:.1f}s")
                    print("=" * 70)
                
                return success
            else:
                # Safe to use segment rotation for non-cross-fade transitions
                self._log_info(f"✅ Transition '{transition_type}' allows segment rotation")
                
                return self.fix_slides_by_segments(rotate_config)
        
        return True
    
    def _parse_arrange_spec(self, spec: str, total_slides: int) -> List[int]:
        """
        Parse arrange specification
        
        Format: "0-8,11,9-10" means slides in order: 0,1,2,3,4,5,6,7,8,11,9,10
        
        Args:
            spec: Arrange specification string
            total_slides: Total number of slides
            
        Returns:
            List of slide indices in new order
            
        Raises:
            ValueError: If format is invalid
        """
        if not spec or not spec.strip():
            return list(range(total_slides))
        
        new_order = []
        parts = spec.split(',')
        
        for part in parts:
            part = part.strip()
            
            if '-' in part:
                # Range
                try:
                    start, end = part.split('-')
                    start, end = int(start), int(end)
                    
                    if start < 0 or end >= total_slides:
                        raise ValueError(f"Range {start}-{end} out of bounds (0-{total_slides-1})")
                    
                    for i in range(start, end + 1):
                        new_order.append(i)
                except ValueError as e:
                    raise ValueError(f"Invalid range '{part}': {e}")
            else:
                # Single slide
                try:
                    slide_idx = int(part)
                    
                    if slide_idx < 0 or slide_idx >= total_slides:
                        raise ValueError(f"Slide {slide_idx} out of bounds (0-{total_slides-1})")
                    
                    new_order.append(slide_idx)
                except ValueError as e:
                    raise ValueError(f"Invalid slide number '{part}': {e}")
        
        # Validate: all slides must be present exactly once
        if len(new_order) != total_slides or len(set(new_order)) != total_slides:
            raise ValueError("Arrange must include all slides exactly once")
        
        return new_order
    
    def fix_slides(self, fix_config: Dict[int, int]) -> bool:
        """
        Fix slide rotations (command-line --fix interface)
        
        Args:
            fix_config: Dict mapping slide indices to rotation degrees
            
        Returns:
            True if successful
        """
        self.start_time = time.time()
        
        self._log_step_start(f"🔧 Fixing {len(fix_config)} slide(s)")
        
        # Load video config
        video_config = self.load_video_config()
        if not video_config:
            return False
        
        total_slides = len(video_config['images'])
        transition_type = video_config.get('transition_type', 'unknown')
        
        self._log_info(f"📊 Total slides: {total_slides}")
        self._log_info(f"🎨 Transition type: {transition_type}")
        
        # Validate slide indices
        invalid_slides = [idx for idx in fix_config.keys() if idx >= total_slides]
        if invalid_slides:
            print(f"❌ Invalid slide indices: {invalid_slides}")
            print(f"   Video has {total_slides} slides (0-{total_slides-1})")
            return False
        
        # Check if full rebuild is required
        needs_rebuild = self.requires_full_rebuild(fix_config, video_config)
        
        if needs_rebuild:
            self._log_info(f"🔄 Cross-fade transition detected: '{transition_type}'")
            self._log_info("   Full video rebuild required to regenerate transitions")
            
            success = self.rebuild_video_with_rotations(fix_config, video_config)
            
            if success:
                total_elapsed = time.time() - self.start_time
                print("\n" + "=" * 70)
                print(f"✅ Video rebuild complete in {total_elapsed:.1f}s")
                print("=" * 70)
            
            return success
        else:
            # Safe to use segment rotation
            self._log_info(f"✅ Transition '{transition_type}' allows segment rotation")
            
            return self.fix_slides_by_segments(fix_config)
    
    def fix_slides_by_segments(self, fix_config: Dict[int, int]) -> bool:
        """
        Fix slides by extracting, rotating, and reassembling segments
        (Fast method for non-cross-fade transitions)
        
        Args:
            fix_config: Dict mapping slide indices to rotation degrees
            
        Returns:
            True if successful
        """
        # Load video config
        video_config = self.load_video_config()
        if not video_config:
            return False
        
        total_slides = len(video_config['images'])
        
        # Extract all slides
        self._log_step_start("📦 Extracting slides")
        slide_segments = []
        
        for i in tqdm(range(total_slides), desc="Extracting", ncols=80, unit="slide"):
            segment_path = os.path.join(self.temp_dir, f'slide_{i:04d}.mp4')
            
            if not self.extract_slide(i, video_config, segment_path):
                print(f"❌ Failed to extract slide {i}")
                return False
            
            slide_segments.append(segment_path)
        
        self._log_step_end(f"Extracted {total_slides} slides")
        
        # Rotate specified slides
        self._log_step_start(f"🔄 Rotating {len(fix_config)} slide(s)")
        
        for slide_idx, degrees in tqdm(fix_config.items(), desc="Rotating", ncols=80):
            if slide_idx >= total_slides:
                continue
            
            original_path = slide_segments[slide_idx]
            rotated_path = os.path.join(self.temp_dir, f'slide_{slide_idx:04d}_rotated.mp4')
            
            if not self.rotate_slide(original_path, rotated_path, degrees):
                print(f"❌ Failed to rotate slide {slide_idx}")
                return False
            
            # Replace original with rotated
            os.remove(original_path)
            os.rename(rotated_path, original_path)
            
            # Update config
            video_config['images'][slide_idx]['custom_rotation'] = degrees
        
        self._log_step_end("Rotations complete")
        
        # Assemble video (without audio)
        self._log_step_start("📹 Assembling video")
        temp_output = self.video_path.replace('.mp4', '_temp_no_audio.mp4')
        
        if not self.assemble_video(slide_segments, temp_output):
            return False
        
        self._log_step_end("Video assembled")
        
        # Update config file
        with open(self.config_path, 'w') as f:
            json.dump(video_config, f, indent=2)
        
        # Restore audio track
        final_output = self.video_path.replace('.mp4', '_temp_final.mp4')
        audio_restored = False
        
        # Try config music first
        music_file = self.config.expand_path('music_file')
        if music_file and os.path.exists(music_file):
            self._log_step_start(f"🎵 Adding audio: {os.path.basename(music_file)}")
            if self._add_audio_track(temp_output, music_file, final_output):
                audio_restored = True
                self._log_step_end("Audio added")
        
        # Fallback to original video audio
        if not audio_restored and os.path.exists(self.video_path):
            self._log_step_start("🎵 Extracting and restoring original audio")
            temp_audio = os.path.join(self.temp_dir, 'original_audio.aac')
            
            extract_cmd = [
                'ffmpeg', '-y',
                '-i', self.video_path,
                '-vn',
                '-acodec', 'copy',
                temp_audio
            ]
            
            try:
                subprocess.run(extract_cmd, check=True, 
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
                
                if self._add_audio_track(temp_output, temp_audio, final_output):
                    audio_restored = True
                    self._log_step_end("Audio restored")
                
                try:
                    os.remove(temp_audio)
                except:
                    pass
                    
            except subprocess.CalledProcessError:
                self._log_info("   ⚠️  No audio found in original video")
        
        # Replace original video
        os.remove(self.video_path)
        
        if audio_restored and os.path.exists(final_output):
            os.rename(final_output, self.video_path)
            try:
                os.remove(temp_output)
            except:
                pass
        else:
            os.rename(temp_output, self.video_path)
            self._log_info("   ℹ️  Video assembled without audio")
        
        # Cleanup temp segments
        self._log_step_start("🧹 Cleaning up temporary files")
        for segment in slide_segments:
            try:
                os.remove(segment)
            except:
                pass
        self._log_step_end("Cleanup complete")
        
        total_elapsed = time.time() - self.start_time
        print("\n" + "=" * 70)
        print(f"✅ Slide fixes complete in {total_elapsed:.1f}s: {self.video_path}")
        print("=" * 70)
        
        return True
    
    def _add_audio_track(self, video_path: str, audio_path: str, 
                        output_path: str) -> bool:
        """
        Add audio track to video
        
        Args:
            video_path: Input video (no audio)
            audio_path: Audio file path
            output_path: Output video with audio
            
        Returns:
            True if successful
        """
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', audio_path,
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            '-movflags', '+faststart',
            output_path
        ]
        
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  Could not add audio: {e}")
            return False


def parse_rotate_spec(spec: str) -> Dict[int, int]:
    """
    Parse rotate specification from config
    
    Format: "1:180,8-23:90" -> {1: 180, 8: 90, 9: 90, ..., 23: 90}
    
    Args:
        spec: Rotate specification string
        
    Returns:
        Dict mapping slide indices to rotation degrees
        
    Raises:
        ValueError: If format is invalid
    """
    if not spec or not spec.strip():
        return {}
    
    rotate_config = {}
    
    # Split by comma
    parts = spec.split(',')
    
    for part in parts:
        part = part.strip()
        
        if ':' not in part:
            raise ValueError(f"Invalid format '{part}' (expected 'slides:degrees')")
        
        slides_part, degrees_part = part.split(':', 1)
        
        # Parse degrees
        try:
            degrees = int(degrees_part)
            if degrees not in [90, 180, 270]:
                raise ValueError(f"Degrees must be 90, 180, or 270 (got {degrees})")
        except ValueError as e:
            raise ValueError(f"Invalid degrees '{degrees_part}': {e}")
        
        # Parse slides
        if '-' in slides_part:
            # Range
            try:
                start, end = slides_part.split('-')
                start, end = int(start), int(end)
                
                for i in range(start, end + 1):
                    rotate_config[i] = degrees
            except ValueError as e:
                raise ValueError(f"Invalid range '{slides_part}': {e}")
        else:
            # Single slide
            try:
                slide_idx = int(slides_part)
                rotate_config[slide_idx] = degrees
            except ValueError as e:
                raise ValueError(f"Invalid slide number '{slides_part}': {e}")
    
    return rotate_config


def parse_slide_spec(spec_str: str) -> Dict[int, int]:
    """
    Parse slide specification string (legacy --fix format)
    
    Formats:
        "8:180" -> {8: 180}
        "1-5:90" -> {1: 90, 2: 90, 3: 90, 4: 90, 5: 90}
        "1,3,5:270" -> {1: 270, 3: 270, 5: 270}
    
    Args:
        spec_str: Specification string
        
    Returns:
        Dict mapping slide indices to rotation degrees
        
    Raises:
        ValueError: If format is invalid
    """
    return parse_rotate_spec(spec_str)


def parse_fix_arguments(fix_args: List[str]) -> Optional[Dict[int, int]]:
    """
    Parse list of fix arguments from command line
    
    Args:
        fix_args: List of specification strings
        
    Returns:
        Combined dict of all fixes or None if error
    """
    fix_config = {}
    
    for arg in fix_args:
        try:
            parsed = parse_rotate_spec(arg)
            fix_config.update(parsed)
        except ValueError as e:
            print(f"❌ Error parsing '{arg}': {e}")
            print("   Valid formats:")
            print("     8:180           - rotate slide 8 by 180°")
            print("     1-5:90          - rotate slides 1-5 by 90°")
            print("     1,3,5:270       - rotate slides 1,3,5 by 270°")
            print("     1-3,5-7:90      - rotate slides 1-3 and 5-7 by 90°")
            return None
    
    return fix_config
