"""
Configuration management for Memories Creator
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional


# Default configuration 
DEFAULT_CONFIG = {
    # Directories 
    "art_dir": "./photos",
    "output_dir": "./output",
    
    # Music settings
    "music_file": "",
    "music_prompt": "acoustic guitar, nostalgic rhythms, style of solo guitar",
    "generate_music": True,
    "music_model": "small",  # small, medium, large, melody
    "music_duration": None,  # Auto-calculated from video
    "music_volume": 0.5,
    "music_fade_in": 2.0,
    "music_fade_out": 6.0,
    "music_seamless": True,
    
    # Video settings
    "max_video_size": 1920,
    "fps": 30,
    "seconds_per_photo": 3,
    
    # Processing
    "max_workers": 8,
    "year_threshold": 2020,
    
    # Transitions
    "transition_type": "pan",
    "transition_intensity": 0.1,
    
    # Image enhancement
    "vignette_strength": 0.25,
    "enhance_old_photos": True,
    "old_photo_enhancement_strength": 0.03,
    
    # Audio encoding
    "audio_bitrate": "192k",
    "audio_codec": "aac",
    
    # Video encoding
    "video_codec": "hevc_videotoolbox",
    "video_quality": 23,
    
    # Format conversion
    "convert_heic": True,
    "convert_heif": True,
    
    # Resize algorithm
    "resize_algorithm": "bicubic",
    
    # Slide editing
    "arrange": "",
    "rotate": "",
}


class Config:
    """Configuration management"""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration
        
        Args:
            config_dict: Custom configuration (uses defaults if None)
        """
        self.config = DEFAULT_CONFIG.copy()
        
        if config_dict:
            # Update with user config, but don't overwrite with None values
            for key, value in config_dict.items():
                if value is not None:
                    self.config[key] = value
        
        # Expand paths
        self._expand_all_paths()
    
    @classmethod
    def from_file(cls, config_path: str = "config.json") -> 'Config':
        """
        Load configuration from JSON file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Config instance
        """
        config_path = os.path.expanduser(config_path)
        
        if not os.path.exists(config_path):
            print(f"⚠️  Config file not found: {config_path}")
            print(f"   Using default configuration")
            return cls()
        
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Filter out comments (keys starting with '_')
            config_dict = {
                k: v for k, v in config_dict.items()
                if not k.startswith('_')
            }
            
            print(f"✅ Loaded configuration from: {config_path}")
            return cls(config_dict)
            
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in config file: {e}")
            print(f"   Using default configuration")
            return cls()
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            print(f"   Using default configuration")
            return cls()
    
    @classmethod
    def create_template(cls, output_path: str = "config.json"):
        """
        Create a template configuration file with helpful comments
        
        Args:
            output_path: Where to save template
        """
        # Start with default config
        template = DEFAULT_CONFIG.copy()
        
        # Add helpful metadata as comments (these will be saved with '_' prefix)
        metadata = {
            "_comment": "Apple Memories Creator Configuration Template",
            "_transition_types": "ken_burns, zoom_in, zoom_out, fade_in, fade_out, fade_cross, dissolve, pan, static",
            "_video_codec_options": "libx264 (CPU), h264_videotoolbox (Apple H.264), hevc_videotoolbox (Apple H.265)",
            "_music_models": "small (fast, ~1.5GB), medium (balanced, ~3.3GB), large (best, ~6.7GB), melody (with melody conditioning)",
            "_resize_algorithm_help": "bilinear (fastest), bicubic (good quality), lanczos (best quality)",
            "_arrange_help": "Reorder slides: '0-8,11,9-10' moves slide 11 after slide 8. Empty = original order",
            "_rotate_help": "Rotate slides: '1:180,8-23:90' rotates slide 1 by 180° and slides 8-23 by 90°. Empty = no rotation",
            "_music_prompt_examples": [
                "uplifting piano melody, emotional strings, cinematic",
                "acoustic guitar, peaceful nature sounds, meditation",
                "epic orchestral, adventure theme, dramatic",
                "jazz piano trio, smooth saxophone, relaxing",
                "electronic ambient, dreamy synth pads, ethereal"
            ]
        }
        
        # Merge metadata with config
        template = {**metadata, **template}
        
        # Ensure output directory exists
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"✅ Configuration template created: {output_path}")
        print(f"   Edit this file and save as 'config.json' in your project root")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value
        
        # Re-expand paths if it's a path key
        if key.endswith('_dir') or key.endswith('_file'):
            self._expand_all_paths()
    
    def expand_path(self, key: str) -> str:
        """
        Expand path for a configuration key
        
        Args:
            key: Configuration key containing a path
            
        Returns:
            Expanded absolute path
        """
        path = self.config.get(key, '')
        if not path:
            return ''
        
        # Expand user home directory
        path = os.path.expanduser(path)
        
        # Make absolute if relative
        if not os.path.isabs(path):
            # Use current working directory as base
            path = os.path.join(os.getcwd(), path)
        
        return os.path.abspath(path)
    
    def _expand_all_paths(self):
        """Pre-expand all path configurations"""
        path_keys = [
            'art_dir',
            'output_dir',
            'music_file',
        ]
        
        for key in path_keys:
            if key in self.config and self.config[key]:
                self.config[f'_{key}_expanded'] = self.expand_path(key)
    
    def should_generate_music(self) -> bool:
        """Check if music should be generated"""
        return (
            self.config.get('generate_music', False) and
            self.config.get('music_prompt', '').strip() != ''
        )
    
    def has_music_file(self) -> bool:
        """Check if music file is specified and exists"""
        music_file = self.config.get('music_file', '').strip()
        if not music_file:
            return False
        
        expanded = self.expand_path('music_file')
        return os.path.exists(expanded)
    
    def get_music_source(self) -> tuple[str, Optional[str]]:
        """
        Determine music source
        
        Returns:
            Tuple of (source_type, source_value)
            source_type: 'generate', 'file', or 'none'
            source_value: prompt for generate, path for file, None for none
        """
        if self.should_generate_music():
            return ('generate', self.config.get('music_prompt'))
        elif self.has_music_file():
            return ('file', self.expand_path('music_file'))
        else:
            return ('none', None)
    
    def print_summary(self):
        """Print configuration summary"""
        print("\n" + "=" * 70)
        print("⚙️  CONFIGURATION SUMMARY")
        print("=" * 70)
        
        # Current working directory
        print(f"\n📂 Current directory: {os.getcwd()}")
        
        # Directories
        print("\n📁 Directories:")
        art_dir = self.expand_path('art_dir')
        output_dir = self.expand_path('output_dir')
        print(f"   Input (art_dir): {art_dir}")
        print(f"   Output (output_dir): {output_dir}")
        
        # Check if directories exist
        if not os.path.exists(art_dir):
            print(f"   ⚠️  Input directory does not exist: {art_dir}")
        if not os.path.exists(output_dir):
            print(f"   📁 Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        # Music
        print("\n🎵 Music:")
        source_type, source_value = self.get_music_source()
        if source_type == 'generate':
            print(f"   Mode: AI Generation")
            print(f"   Prompt: {source_value}")
            print(f"   Model: {self.get('music_model', 'small')}")
            print(f"   Seamless: {self.get('music_seamless', True)}")
        elif source_type == 'file':
            print(f"   Mode: File")
            print(f"   File: {source_value}")
        else:
            print(f"   Mode: None (silent video)")
        
        if source_type != 'none':
            print(f"   Volume: {self.get('music_volume', 0.5) * 100:.0f}%")
            print(f"   Fade in: {self.get('music_fade_in', 2.0)}s")
            print(f"   Fade out: {self.get('music_fade_out', 6.0)}s")
        
        # Video
        print("\n🎬 Video:")
        print(f"   Max size: {self.get('max_video_size')}px")
        print(f"   FPS: {self.get('fps')}")
        print(f"   Duration per photo: {self.get('seconds_per_photo')}s")
        print(f"   Codec: {self.get('video_codec')}")
        print(f"   Quality: {self.get('video_quality')}")
        
        # Transitions
        print("\n✨ Transitions:")
        print(f"   Type: {self.get('transition_type')}")
        print(f"   Intensity: {self.get('transition_intensity')}")
        
        # Enhancement
        print("\n🖼️  Enhancement:")
        print(f"   Old photos (<{self.get('year_threshold')}): ", end='')
        if self.get('enhance_old_photos'):
            strength = self.get('old_photo_enhancement_strength')
            print(f"Enhanced ({strength * 100:.1f}%)")
        else:
            print("No enhancement")
        print(f"   Vignette: {self.get('vignette_strength') * 100:.0f}%")
        print(f"   Resize: {self.get('resize_algorithm')}")
        
        # Slide edits
        arrange = self.get('arrange', '').strip()
        rotate = self.get('rotate', '').strip()
        if arrange or rotate:
            print("\n📝 Slide Edits:")
            if arrange:
                print(f"   Arrange: {arrange}")
            if rotate:
                print(f"   Rotate: {rotate}")
        
        print("\n" + "=" * 70 + "\n")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding comments)"""
        return {
            k: v for k, v in self.config.items()
            if not k.startswith('_')
        }
    
    def save(self, output_path: str = "config.json"):
        """
        Save configuration to file
        
        Args:
            output_path: Where to save (default: config.json in current directory)
        """
        # Use absolute path
        output_path = os.path.abspath(output_path)
        
        # Get config without internal keys
        config_to_save = self.to_dict()
        
        # Add helpful comments (from create_template method)
        final_config = {
            "_comment": "Memories Creator Configuration Template",
            "_transition_types": "ken_burns, zoom_in, zoom_out, fade_in, fade_out, fade_cross, dissolve, pan, static",
            "_video_codec_options": "libx264 (CPU), h264_videotoolbox (Apple H.264), hevc_videotoolbox (Apple H.265)",
            "_music_models": "small (fast, ~1.5GB), medium (balanced, ~3.3GB), large (best, ~6.7GB), melody (with melody conditioning)",
            "_resize_algorithm_help": "bilinear (fastest), bicubic (good quality), lanczos (best quality)",
            "_arrange_help": "Reorder slides: '0-8,11,9-10' moves slide 11 after slide 8. Empty = original order",
            "_rotate_help": "Rotate slides: '1:180,8-23:90' rotates slide 1 by 180° and slides 8-23 by 90°. Empty = no rotation",
            "_music_prompt_examples": [
                "uplifting piano melody, emotional strings, cinematic",
                "acoustic guitar, peaceful nature sounds, meditation",
                "epic orchestral, adventure theme, dramatic",
                "jazz piano trio, smooth saxophone, relaxing",
                "electronic ambient, dreamy synth pads, ethereal"
            ],
            **config_to_save
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(final_config, f, indent=2)
            
            print(f"✅ Configuration saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")
    
    def save_to_project_root(self):
        """
        Save configuration to project root (where the script is running from)
        
        This ensures the config is saved in the right location
        """
        # Save to current working directory
        config_path = os.path.join(os.getcwd(), "config.json")
        self.save(config_path)


def load_config(config_path: str = "config.json") -> Config:
    """
    Convenience function to load configuration
    
    Args:
        config_path: Path to config file
        
    Returns:
        Config instance
    """
    return Config.from_file(config_path)


def create_default_config():
    """Create a default config.json in the current directory"""
    config = Config()
    config.save_to_project_root()
    return config