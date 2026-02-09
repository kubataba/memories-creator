"""
Memories Creator - Source Module
Professional video slideshow creation with Apple-style effects and AI music
"""

__version__ = '1.0.0'
__author__ = 'Memories Creator Team'
__description__ = 'Professional video slideshow creation with Apple-style transitions and AI-generated music'

# Version info accessible via attribute
version = __version__

# Import main components
try:
    from .config import Config, DEFAULT_CONFIG, load_config
    from .transitions import TransitionEngine, TorchTransitionEngine
    from .media_utils import MediaConverter, ExifParser, ImageAnalyzer, ImageProcessor
    from .video_processor import VideoProcessor
    from .slide_editor import SlideEditor, parse_slide_spec, parse_fix_arguments
    from .music_generator import MusicGenerator, generate_music, load_music, MUSIC_GENERATION_AVAILABLE
except ImportError:
    # Allow imports when package is not fully installed
    pass

__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__description__',
    'version',
    
    # Config
    'Config',
    'DEFAULT_CONFIG',
    'load_config',
    
    # Transitions
    'TransitionEngine',
    'TorchTransitionEngine',
    
    # Media utilities
    'MediaConverter',
    'ExifParser',
    'ImageAnalyzer',
    'ImageProcessor',
    
    # Video processing
    'VideoProcessor',
    
    # Slide editing
    'SlideEditor',
    'parse_slide_spec',
    'parse_fix_arguments',
    
    # Music generation
    'MusicGenerator',
    'generate_music',
    'load_music',
    'MUSIC_GENERATION_AVAILABLE',
]