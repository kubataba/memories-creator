#!/usr/bin/env python3
"""
Memories Style Slideshow Creator
Professional video slideshow creation with Apple-style transitions
Optimized for Apple Silicon (M1/M2/M3/M4) with MPS acceleration
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Optional
import torch
from tqdm import tqdm

from .config import Config
from .media_utils import MediaConverter, ImageAnalyzer
from .video_processor import VideoProcessor
from .slide_editor import SlideEditor, parse_fix_arguments
from .music_generator import MusicGenerator, MUSIC_GENERATION_AVAILABLE


class MemoriesCreator:
    """Main application class"""
    
    def __init__(self, config: Config):
        """Initialize with configuration"""
        self.config = config
        
        # Setup paths
        self.art_dir = config.expand_path('art_dir')
        self.output_dir = config.expand_path('output_dir')
        self.music_file = config.expand_path('music_file')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup device
        self.use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
        self.device = torch.device('mps' if self.use_mps else 'cpu')
        
        # Initialize video processor
        self.video_processor = VideoProcessor(config, self.device)
        
        # Initialize music generator if needed
        self.music_generator = None
        if config.should_generate_music():
            if MUSIC_GENERATION_AVAILABLE:
                self.music_generator = MusicGenerator(
                    model_name=config.get('music_model', 'small')
                )
            else:
                print("⚠️  Music generation not available - install dependencies")
        
        # File paths (all relative to current directory)
        self.video_config_file = os.path.join(self.output_dir, 'video_config.json')
        self.temp_video = os.path.join(self.output_dir, 'temp_video.mp4')
        self.final_video = os.path.join(self.output_dir, 'apple_memories.mp4')
    
    def print_header(self):
        """Print application header"""
        print("\n" + "=" * 70)
        print("🍎  Apple Memories Style Slideshow Creator")
        print("   Professional video creation optimized for Apple Silicon")
        print("=" * 70)
        
        # Show device info
        if self.use_mps:
            print(f"✅ Using MPS acceleration (Apple Silicon)")
        else:
            print(f"⚠️  MPS not available, using CPU")
        
        # Show video mode
        transition_type = self.config.get('transition_type', 'ken_burns')
        if transition_type == 'static':
            print(f"🎬 Mode: STATIC (fast, no transitions)")
        elif transition_type == 'pan':
            print(f"🎬 Mode: ENHANCED PAN (movement + gentle zoom)")
        else:
            print(f"🎬 Mode: {transition_type.upper()} (GPU transitions)")
        
        # Show optimizations
        codec = self.config.get('video_codec', 'libx264')
        if codec in ['h264_videotoolbox', 'hevc_videotoolbox']:
            print(f"⚡ Codec: {codec} (Apple VideoToolbox hardware acceleration)")
        else:
            print(f"⚡ Codec: {codec}")
        
        # Show resize algorithm
        resize_algo = self.config.get('resize_algorithm', 'bicubic')
        print(f"🔍 Resize algorithm: {resize_algo}")
        
        # Show config edits status
        arrange_spec = self.config.get('arrange', '').strip()
        rotate_spec = self.config.get('rotate', '').strip()
        if arrange_spec or rotate_spec:
            print(f"📝 Config edits: Will be applied after creation")
            if arrange_spec:
                print(f"   Arrange: {arrange_spec}")
            if rotate_spec:
                print(f"   Rotate: {rotate_spec}")
        
        # Show music status
        source_type, source_value = self.config.get_music_source()
        if source_type == 'generate':
            print(f"🎵 Music: AI Generation")
            print(f"   Prompt: {source_value}")
        elif source_type == 'file':
            print(f"🎵 Music: {os.path.basename(source_value)}")
        else:
            print(f"🎵 Music: None (silent video)")
        
        print(f"📁 Current directory: {os.getcwd()}")
        print()
    
    def scan_images(self) -> List[str]:
        """Scan art directory for images"""
        print(f"📁 Scanning directory: {self.art_dir}")
        
        if not os.path.exists(self.art_dir):
            print(f"❌ Directory not found: {self.art_dir}")
            print(f"   Please create the directory or update config.art_dir")
            print(f"   Current config.art_dir: {self.config.get('art_dir')}")
            return []
        
        image_files = []
        
        for file in sorted(os.listdir(self.art_dir)):
            filepath = os.path.join(self.art_dir, file)
            
            if not os.path.isfile(filepath):
                continue
            
            if MediaConverter.is_supported(filepath):
                image_files.append(filepath)
        
        print(f"   Found {len(image_files)} supported images")
        return image_files
    
    def convert_images(self, image_files: List[str]) -> List[str]:
        """Convert non-standard formats to JPEG"""
        converted_files = []
        to_convert = []
        
        for filepath in image_files:
            if MediaConverter.needs_conversion(filepath):
                to_convert.append(filepath)
            else:
                converted_files.append(filepath)
        
        if not to_convert:
            return image_files
        
        print(f"\n🔄 Converting {len(to_convert)} non-standard formats...")
        
        conversion_dir = os.path.join(self.output_dir, 'converted')
        os.makedirs(conversion_dir, exist_ok=True)
        
        for filepath in tqdm(to_convert, desc="Converting", ncols=80, unit="img"):
            converted = MediaConverter.convert_to_jpeg(filepath, conversion_dir)
            if converted:
                converted_files.append(converted)
            else:
                print(f"⚠️  Skipping {os.path.basename(filepath)} - conversion failed")
        
        return sorted(converted_files)
    
    def analyze_images(self, image_files: List[str]) -> List[Dict]:
        """Analyze all images and apply custom rotations from config"""
        print("\n🔍 Analyzing images...")
        
        images_info = []
        year_threshold = self.config.get('year_threshold', 2020)
        
        # Load rotations from config if exists
        rotation_map = {}
        config_path = os.path.join(self.output_dir, 'video_config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    for img_config in config_data.get('images', []):
                        filename = img_config.get('filename', '')
                        rotation = img_config.get('custom_rotation', 0)
                        if rotation != 0:
                            rotation_map[filename] = rotation
                if rotation_map:
                    print(f"   Loaded {len(rotation_map)} custom rotations from config")
            except:
                pass
        
        for filepath in tqdm(image_files, desc="Analyzing", ncols=80, unit="img"):
            info = ImageAnalyzer.analyze_image(filepath, year_threshold)
            if info:
                # Apply custom rotation if exists in config
                filename = os.path.basename(filepath)
                if filename in rotation_map:
                    info['custom_rotation'] = rotation_map[filename]
                    print(f"   Applying {rotation_map[filename]}° rotation to {filename}")
                
                images_info.append(info)
            else:
                print(f"⚠️  Warning: Skipping {os.path.basename(filepath)} - could not analyze")
        
        if not images_info:
            print("❌ No valid images found")
            return []
        
        # Sort images: old first, horizontal first, by year, by path
        images_info.sort(key=lambda x: (
            x['is_old'],
            0 if x['orientation'] == 'horizontal' else 1,
            x['year'],
            x['path']
        ))
        
        # Print statistics
        old_count = sum(1 for img in images_info if img['is_old'])
        new_count = len(images_info) - old_count
        
        print(f"\n📊 Statistics:")
        print(f"   Total images: {len(images_info)}")
        print(f"   Old photos (<{year_threshold}): {old_count}")
        print(f"   Recent photos: {new_count}")
        
        return images_info

    def calculate_dimensions(self, images_info: List[Dict]) -> tuple:
        """Calculate optimal video dimensions based on image sizes"""
        max_w, max_h = 0, 0
        
        for info in images_info:
            w, h = info['dimensions']
            max_w = max(max_w, w)
            max_h = max(max_h, h)
        
        max_size = self.config.get('max_video_size', 1920)
        video_w = min(max_w, max_size)
        video_h = min(max_h, max_size)
        
        print(f"\n📐 Video dimensions: {video_w}x{video_h}")
        return video_w, video_h
    
    def save_config(self, images_info: List[Dict], video_w: int, video_h: int):
        """Save video configuration for editing and rotation"""
        config_data = {
            'video_width': video_w,
            'video_height': video_h,
            'fps': self.config.get('fps'),
            'seconds_per_photo': self.config.get('seconds_per_photo'),
            'transition_type': self.config.get('transition_type'),
            'video_codec': self.config.get('video_codec'),
            'images': []
        }
        
        for i, img_info in enumerate(images_info):
            config_data['images'].append({
                'index': i,
                'path': img_info['path'],
                'filename': img_info['filename'],
                'custom_rotation': img_info.get('custom_rotation', 0),
                'duration_seconds': self.config.get('seconds_per_photo')
            })
        
        with open(self.video_config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"✅ Configuration saved: {self.video_config_file}")
    
    def create_video(self):
        """Main video creation workflow"""
        self.print_header()
        
        # 1. Scan for images
        image_files = self.scan_images()
        if not image_files:
            print(f"\n💡 Tip: Create a folder called 'art' in the current directory")
            print(f"       or update config.art_dir in config.json")
            return
        
        # 2. Convert non-standard formats
        if self.config.get('convert_heic') or self.config.get('convert_heif'):
            image_files = self.convert_images(image_files)
        
        # 3. Calculate video dimensions from sample
        print("\n📏 Calculating optimal video dimensions...")
        temp_info = []
        for filepath in image_files[:10]:  # Sample first 10 images
            info = ImageAnalyzer.analyze_image(filepath, self.config.get('year_threshold', 2020))
            if info:
                temp_info.append(info)
        
        if not temp_info:
            print("❌ Could not analyze images")
            return
        
        video_w, video_h = self.calculate_dimensions(temp_info)
        
        # 4. Full image analysis
        images_info = self.analyze_images(image_files)
        if not images_info:
            return
        
        # 5. Create video
        success = self.video_processor.create_video(
            images_info, video_w, video_h, self.temp_video
        )
        
        if not success:
            print("❌ Video creation failed")
            return
        
        # 6. Handle music
        print("\n" + "=" * 70)
        print("🎵 Processing Audio")
        print("=" * 70)
        
        source_type, source_value = self.config.get_music_source()
        prepared_music_file = None
        
        if source_type == 'generate':
            # Generate music from prompt
            print(f"\n🎼 Generating AI music...")
            video_duration = len(images_info) * self.config.get('seconds_per_photo')
            
            try:
                music_file = self.music_generator.generate(
                    prompt=source_value,
                    duration=video_duration,
                    output_path=os.path.join(self.output_dir, 'generated_music.wav')
                )
                
                # Prepare for video
                prepared_music_file = self.music_generator.prepare_for_video(
                    audio_source=music_file,
                    video_duration=video_duration,
                    volume=self.config.get('music_volume', 0.3),
                    fade_in=self.config.get('music_fade_in', 2.0),
                    fade_out=self.config.get('music_fade_out', 3.0),
                    output_path=os.path.join(self.output_dir, 'prepared_music.wav')
                )
            except Exception as e:
                print(f"⚠️  Music generation failed: {e}")
                print("   Continuing with silent video")
        
        elif source_type == 'file':
            # Use existing music file
            video_duration = len(images_info) * self.config.get('seconds_per_photo')
            
            try:
                if self.music_generator:
                    # Prepare music for video
                    prepared_music_file = self.music_generator.prepare_for_video(
                        audio_source=source_value,
                        video_duration=video_duration,
                        volume=self.config.get('music_volume', 0.3),
                        fade_in=self.config.get('music_fade_in', 2.0),
                        fade_out=self.config.get('music_fade_out', 3.0),
                        output_path=os.path.join(self.output_dir, 'prepared_music.wav')
                    )
                else:
                    prepared_music_file = source_value
            except Exception as e:
                print(f"⚠️  Music preparation failed: {e}")
                prepared_music_file = source_value
        
        # Add audio to video
        if prepared_music_file and os.path.exists(prepared_music_file):
            self.video_processor.add_audio(
                self.temp_video, prepared_music_file, self.final_video
            )
            # Clean up temp file
            if os.path.exists(self.temp_video):
                os.remove(self.temp_video)
        else:
            # No audio - rename temp to final
            os.rename(self.temp_video, self.final_video)
        
        # 7. Save configuration
        self.save_config(images_info, video_w, video_h)
        
        # 8. Check if config has arrange/rotate edits and apply them
        arrange_spec = self.config.get('arrange', '').strip()
        rotate_spec = self.config.get('rotate', '').strip()
        
        if arrange_spec or rotate_spec:
            print("\n" + "=" * 70)
            print("📝 Config has slide edits - applying automatically...")
            print("=" * 70)
            
            editor = SlideEditor(self.config, self.final_video, self.video_config_file)
            
            if editor.apply_config_edits():
                print("\n✅ Config edits applied successfully!")
            else:
                print("\n⚠️  Could not apply config edits")
        
        # 9. Show results
        if os.path.exists(self.final_video):
            file_size = os.path.getsize(self.final_video) / (1024 * 1024)
            duration = len(images_info) * self.config.get('seconds_per_photo')
            
            print("\n" + "=" * 70)
            print("✨ Video Created Successfully!")
            print("=" * 70)
            print(f"📹 Output: {self.final_video}")
            print(f"📐 Resolution: {video_w}x{video_h}")
            print(f"📊 File size: {file_size:.1f} MB")
            print(f"⏱️  Duration: {duration // 60}m {duration % 60}s")
            print(f"📸 Photos: {len(images_info)}")
            
            transition_type = self.config.get('transition_type', 'ken_burns')
            if transition_type == 'static':
                print(f"🎨 Transition: None (static, fastest)")
            elif transition_type == 'pan':
                print(f"🎨 Transition: Enhanced Pan (movement + zoom)")
            else:
                print(f"🎨 Transition: {transition_type}")
            
            codec = self.config.get('video_codec', 'libx264')
            print(f"⚡ Encoding: {codec}")
            
            if arrange_spec or rotate_spec:
                print(f"📝 Config edits: Applied")
            
            print("=" * 70 + "\n")
            
            # Open in Finder
            os.system(f'open -R "{self.final_video}"')
    
    def replace_audio(self, audio_file: str):
        """Replace audio in existing video"""
        self.print_header()
        
        if not os.path.exists(self.final_video):
            print(f"❌ Video not found: {self.final_video}")
            print("   Create video first without --audio flag")
            return
        
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            return
        
        output_path = self.final_video.replace('.mp4', '_with_audio.mp4')
        
        success = self.video_processor.replace_audio(
            self.final_video, audio_file, output_path
        )
        
        if success:
            # Replace original
            os.remove(self.final_video)
            os.rename(output_path, self.final_video)
            print(f"\n✅ Audio replaced: {self.final_video}")
            os.system(f'open -R "{self.final_video}"')
    
    def list_slides(self):
        """Display list of all slides"""
        if not os.path.exists(self.video_config_file):
            print("❌ No video configuration found. Create video first.")
            return
        
        with open(self.video_config_file, 'r') as f:
            config = json.load(f)
        
        print(f"\n📋 Slide List (Total: {len(config['images'])})")
        print("=" * 80)
        
        for img in config['images']:
            idx = img['index']
            filename = img['filename']
            rotation = img.get('custom_rotation', 0)
            rotation_str = f" [rotation: {rotation}°]" if rotation else ""
            
            print(f"  {idx:3d}. {filename}{rotation_str}")
        
        print("\n" + "=" * 80)
        print("\n💡 Examples:")
        print("   python main.py --fix 8:180")
        print("   python main.py --fix 1-5:90")
        print("   python main.py --fix 1,3,5:270")
        print("   python main.py --fix 8:180 12:90")
        print("\n💡 Or edit config.json:")
        print('   "rotate": "1:180,8-23:90"')
        print('   "arrange": "0-8,11,9-10"')
    
    def fix_slides(self, fix_args: List[str]):
        """Fix slide rotations"""
        self.print_header()
        
        # Parse fix arguments
        fix_config = parse_fix_arguments(fix_args)
        if fix_config is None:
            return
        
        if not fix_config:
            print("❌ No slides specified to fix")
            return
        
        # Create slide editor
        editor = SlideEditor(self.config, self.final_video, self.video_config_file)
        
        # Perform fixes
        success = editor.fix_slides(fix_config)
        
        if success:
            print("\n✅ Slides fixed successfully!")
            os.system(f'open -R "{self.final_video}"')
    
    def apply_config_edits(self):
        """Apply slide edits from config.json"""
        self.print_header()
        
        if not os.path.exists(self.final_video):
            print("❌ Video not found. Create video first.")
            return
        
        # Create slide editor
        editor = SlideEditor(self.config, self.final_video, self.video_config_file)
        
        # Apply edits
        success = editor.apply_config_edits()
        
        if success:
            print("\n✅ Config edits applied!")
            os.system(f'open -R "{self.final_video}"')


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Create Apple Memories style slideshows',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Create new video:
    python main.py
  
  Replace audio track:
    python main.py --audio ~/Music/song.mp3
  
  List all slides:
    python main.py --list
  
  Fix slide rotations:
    python main.py --fix 8:180
    python main.py --fix 1-5:90
  
  Apply config.json edits:
    python main.py --apply-config
    (Or just edit config.json and run: python main.py)
  
  Show current settings:
    python main.py --config
        """
    )
    
    parser.add_argument('--config', action='store_true',
                       help='Show current configuration')
    parser.add_argument('--list', action='store_true',
                       help='List all slides in video')
    parser.add_argument('--audio', metavar='FILE',
                       help='Replace audio track with specified file')
    parser.add_argument('--fix', nargs='+', metavar='SPEC',
                       help='Fix slide rotations (e.g., 8:180 or 1-5:90)')
    parser.add_argument('--apply-config', action='store_true',
                       help='Apply arrange/rotate edits from config.json')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Handle commands
    if args.config:
        config.print_summary()
        return
    
    # Create app instance
    app = MemoriesCreator(config)
    
    if args.list:
        app.list_slides()
    elif args.audio:
        audio_path = os.path.expanduser(args.audio)
        app.replace_audio(audio_path)
    elif args.fix:
        app.fix_slides(args.fix)
    elif args.apply_config:
        app.apply_config_edits()
    else:
        app.create_video()


if __name__ == '__main__':
    main()
