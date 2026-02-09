"""
Command-line interface for Memories Creator
Professional video slideshow creation with Apple-style transitions and AI music
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

# Import modules - will work both as script and as installed package
try:
    # Try package import first (when installed)
    from memories_creator.config import Config
    from memories_creator.music_generator import MusicGenerator, MUSIC_GENERATION_AVAILABLE
except ImportError:
    # Fall back to direct import (when running from src/)
    try:
        from config import Config
        from music_generator import MusicGenerator, MUSIC_GENERATION_AVAILABLE
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you have installed the package:")
        print("   pip install -e .")
        sys.exit(1)


def create_video_command(args):
    """Handle video creation command"""
    # Import here to avoid circular imports
    try:
        from memories_creator.main import MemoriesCreator
    except ImportError:
        from main import MemoriesCreator
    
    # Load or create configuration
    config_path = args.config or "config.json"
    
    if os.path.exists(config_path):
        config = Config.from_file(config_path)
        print(f"📋 Loaded configuration from: {config_path}")
    else:
        print(f"📋 Config file not found: {config_path}")
        print("   Creating default configuration...")
        config = Config()
        config.save()
        config.print_summary()
    
    # Override config with command-line arguments
    if args.input:
        config.set('art_dir', args.input)
        print(f"   Override: art_dir = {args.input}")
    if args.output:
        config.set('output_dir', args.output)
        print(f"   Override: output_dir = {args.output}")
    if args.music:
        config.set('music_file', args.music)
        config.set('generate_music', False)
        print(f"   Override: using music file {args.music} (AI generation disabled)")
    if args.transition:
        config.set('transition_type', args.transition)
        print(f"   Override: transition_type = {args.transition}")
    if args.duration:
        config.set('seconds_per_photo', args.duration)
        print(f"   Override: seconds_per_photo = {args.duration}")
    if args.music_model:
        config.set('music_model', args.music_model)
        print(f"   Override: music_model = {args.music_model}")
    if args.music_seamless is not None:
        config.set('music_seamless', args.music_seamless)
        print(f"   Override: music_seamless = {args.music_seamless}")
    
    # Display final configuration
    print("\n🎬 Final configuration for video creation:")
    config.print_summary()
    
    # Create and run
    creator = MemoriesCreator(config)
    creator.create_video()


def music_command(args):
    """Handle music generation command"""
    if not MUSIC_GENERATION_AVAILABLE:
        print("❌ Music generation not available")
        print("   Install dependencies: pip install -r requirements-music.txt")
        sys.exit(1)
    
    # Load or create configuration
    config_path = args.config or "config.json"
    
    if os.path.exists(config_path):
        config = Config.from_file(config_path)
        print(f"📋 Loaded configuration from: {config_path}")
    else:
        print(f"📋 Config file not found: {config_path}")
        print("   Creating default configuration...")
        config = Config()
        config.save()
    
    # Override config with command-line arguments
    if args.model and args.model != 'small':
        config.set('music_model', args.model)
        print(f"   Override: music_model = {args.model}")
    
    if args.music_seamless is not None:
        config.set('music_seamless', args.music_seamless)
        print(f"   Override: music_seamless = {args.music_seamless}")
    
    # Display configuration
    print("\n🎵 Music generation configuration:")
    config.print_summary()
    
    # Create generator with config
    generator = MusicGenerator(config=config, model_name=args.model)
    
    # Generate music
    output_path = args.output or None
    music_path = generator.generate(
        prompt=args.prompt,
        duration=args.duration,
        output_path=output_path,
        seamless=args.music_seamless
    )
    
    print(f"\n✅ Music saved to: {music_path}")
    
    # Display statistics
    generator.print_statistics()
    
    # Play if requested
    if args.play:
        print("\n▶️  Playing generated music...")
        os.system(f'afplay "{music_path}"')


def list_command(args):
    """Handle list slides command"""
    try:
        from memories_creator.main import MemoriesCreator
    except ImportError:
        from main import MemoriesCreator
    
    # Load configuration
    config_path = args.config or "config.json"
    
    if os.path.exists(config_path):
        config = Config.from_file(config_path)
        print(f"📋 Loaded configuration from: {config_path}")
    else:
        print(f"❌ Config file not found: {config_path}")
        print("   Please create a config file first: memories config create")
        sys.exit(1)
    
    config.print_summary()
    creator = MemoriesCreator(config)
    creator.list_slides()


def fix_command(args):
    """Handle fix slides command"""
    try:
        from memories_creator.main import MemoriesCreator
    except ImportError:
        from main import MemoriesCreator
    
    # Load configuration
    config_path = args.config or "config.json"
    
    if os.path.exists(config_path):
        config = Config.from_file(config_path)
        print(f"📋 Loaded configuration from: {config_path}")
    else:
        print(f"❌ Config file not found: {config_path}")
        print("   Please create a config file first: memories config create")
        sys.exit(1)
    
    config.print_summary()
    creator = MemoriesCreator(config)
    creator.fix_slides(args.specs)


def audio_command(args):
    """Handle audio replacement command"""
    try:
        from memories_creator.main import MemoriesCreator
    except ImportError:
        from main import MemoriesCreator
    
    # Load configuration
    config_path = args.config or "config.json"
    
    if os.path.exists(config_path):
        config = Config.from_file(config_path)
        print(f"📋 Loaded configuration from: {config_path}")
    else:
        print(f"❌ Config file not found: {config_path}")
        print("   Please create a config file first: memories config create")
        sys.exit(1)
    
    config.print_summary()
    creator = MemoriesCreator(config)
    creator.replace_audio(args.file)


def config_command(args):
    """Handle configuration commands"""
    if args.action == 'show':
        config_path = args.file or "config.json"
        
        if os.path.exists(config_path):
            config = Config.from_file(config_path)
            config.print_summary()
        else:
            print(f"❌ Config file not found: {config_path}")
            print("   Create one with: memories config create")
            sys.exit(1)
    
    elif args.action == 'create':
        output_path = args.output or "config.json"
        
        if os.path.exists(output_path):
            print(f"⚠️  Config file already exists: {output_path}")
            response = input("   Overwrite? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("   Cancelled.")
                sys.exit(0)
        
        Config.create_template(output_path)
        
        # Show the created config
        if os.path.exists(output_path):
            print(f"\n📋 Created configuration file: {output_path}")
            config = Config.from_file(output_path)
            config.print_summary()
            print(f"\n💡 Edit this file to customize settings, then run:")
            print(f"   memories create")
    
    elif args.action == 'validate':
        config_path = args.file or "config.json"
        
        if os.path.exists(config_path):
            config = Config.from_file(config_path)
            print("✅ Configuration is valid")
            config.print_summary()
        else:
            print(f"❌ Config file not found: {config_path}")
            sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Apple Memories Creator - Professional video slideshows with AI music',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Create a configuration file first:
    memories config create
  
  Create video with default config:
    memories create
  
  Create with custom input/output:
    memories create --input ./photos --output ./videos
  
  Generate AI music:
    memories music --prompt "uplifting piano" --duration 120
  
  Generate music with advanced settings:
    memories music "epic orchestral" --duration 180 --model medium --no-music-seamless
  
  List all slides:
    memories list
  
  Fix slide rotations:
    memories fix 8:180 12:90
  
  Show configuration:
    memories config show
  
  Validate configuration:
    memories config validate
        """
    )
    
    # Global arguments
    parser.add_argument(
        '--version',
        action='store_true',
        help='Show version information'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create command
    create_parser = subparsers.add_parser(
        'create',
        help='Create video from photos'
    )
    create_parser.add_argument(
        '--config', '-c',
        help='Configuration file (default: config.json)'
    )
    create_parser.add_argument(
        '--input', '-i',
        help='Input directory with photos (overrides art_dir in config)'
    )
    create_parser.add_argument(
        '--output', '-o',
        help='Output directory (overrides output_dir in config)'
    )
    create_parser.add_argument(
        '--music', '-m',
        help='Music file to use (disables AI music generation)'
    )
    create_parser.add_argument(
        '--transition', '-t',
        choices=['ken_burns', 'zoom_in', 'zoom_out', 'fade_in', 'fade_out', 
                'fade_cross', 'dissolve', 'pan', 'static'],
        help='Transition type (overrides config)'
    )
    create_parser.add_argument(
        '--duration', '-d',
        type=int,
        help='Seconds per photo (overrides config)'
    )
    create_parser.add_argument(
        '--music-model',
        choices=['small', 'medium', 'large', 'melody'],
        help='AI music model to use (overrides config)'
    )
    create_parser.add_argument(
        '--music-seamless',
        action=argparse.BooleanOptionalAction,
        help='Generate seamless long music (overrides config)'
    )
    create_parser.add_argument(
        '--no-audio',
        action='store_true',
        help='Create video without audio'
    )
    create_parser.set_defaults(func=create_video_command)
    
    # Music command
    music_parser = subparsers.add_parser(
        'music',
        help='Generate AI music'
    )
    music_parser.add_argument(
        'prompt',
        help='Music description prompt (e.g., "uplifting piano melody, emotional strings")'
    )
    music_parser.add_argument(
        '--duration',
        type=float,
        default=60.0,
        help='Duration in seconds (default: 60)'
    )
    music_parser.add_argument(
        '--output', '-o',
        help='Output file path (default: output/music/music_*.wav)'
    )
    music_parser.add_argument(
        '--model',
        choices=['small', 'medium', 'large', 'melody'],
        default='small',
        help='Model size (default: small)'
    )
    music_parser.add_argument(
        '--config', '-c',
        help='Configuration file to load settings from'
    )
    music_parser.add_argument(
        '--play', '-p',
        action='store_true',
        help='Play after generation (macOS only)'
    )
    music_parser.add_argument(
        '--music-seamless',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Generate seamless long music (default: True)'
    )
    music_parser.set_defaults(func=music_command)
    
    # List command
    list_parser = subparsers.add_parser(
        'list',
        help='List all slides in video'
    )
    list_parser.add_argument(
        '--config', '-c',
        help='Configuration file (default: config.json)'
    )
    list_parser.set_defaults(func=list_command)
    
    # Fix command
    fix_parser = subparsers.add_parser(
        'fix',
        help='Fix slide rotations'
    )
    fix_parser.add_argument(
        'specs',
        nargs='+',
        help='Rotation specs (e.g., 8:180 or 1-5:90)'
    )
    fix_parser.add_argument(
        '--config', '-c',
        help='Configuration file (default: config.json)'
    )
    fix_parser.set_defaults(func=fix_command)
    
    # Audio command
    audio_parser = subparsers.add_parser(
        'audio',
        help='Replace audio track in existing video'
    )
    audio_parser.add_argument(
        '--file', '-f',
        required=True,
        help='New audio file'
    )
    audio_parser.add_argument(
        '--config', '-c',
        help='Configuration file (default: config.json)'
    )
    audio_parser.set_defaults(func=audio_command)
    
    # Config command
    config_parser = subparsers.add_parser(
        'config',
        help='Configuration management'
    )
    config_parser.add_argument(
        'action',
        choices=['show', 'create', 'validate'],
        help='Config action'
    )
    config_parser.add_argument(
        '--file', '-f',
        help='Config file (default: config.json)'
    )
    config_parser.add_argument(
        '--output', '-o',
        help='Output file for create action'
    )
    config_parser.set_defaults(func=config_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle version flag - FIXED: No relative import
    if args.version:
        # Show version without complex imports
        print("Memories Creator v1.0.0")
        sys.exit(0)
    
    # Show help if no command
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    # Execute command
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("   This might be a package setup issue.")
        print("   Try reinstalling: pip install -e .")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()