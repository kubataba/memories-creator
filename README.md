# 🍎 Memories Creator

**Professional video slideshow creation with Apple-style transitions and AI-generated music, optimized for Apple Silicon.**

Transform your photo collections into stunning cinematic videos with AI-powered music generation, professional transitions, and automatic photo enhancement—all from the command line or Python API.

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-CC--BY--NC--4.0-orange.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![PyPI](https://img.shields.io/pypi/v/memories-creator.svg)](https://pypi.org/project/memories-creator/)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)](https://github.com/kubataba/memories-creator)

---

## 🌟 Features

**The only open-source slideshow tool with built-in AI music generation.**

### 🎬 Professional Video Creation
- **9 Transition Types**: Ken Burns, zoom in/out, pan, fade in/out, cross-fade, dissolve, static
- **Smart Photo Enhancement**: Automatic enhancement for photos taken before 2020
- **Hardware Acceleration**: Native MPS support for Apple Silicon (M1/M2/M3/M4)
- **Multiple Codecs**: H.264, H.265 (HEVC) with VideoToolbox hardware encoding

### 🎵 AI Music Generation
- **Text-to-Music**: Generate custom soundtracks from natural language descriptions
- **Multiple Models**: Small (fast), Medium (balanced), Large (best quality), Melody (melody-aware)
- **Seamless Looping**: Intelligent audio extension for any video length
- **Professional Mixing**: Automatic volume balancing, fade in/out, crossfades

### ⚡ Performance
- **GPU Acceleration**: PyTorch MPS backend for transitions and effects
- **Multi-threaded**: Parallel image processing
- **Memory Efficient**: Handles 100+ photos with 16GB RAM

---

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install memories-creator
```

### From Source

```bash
git clone https://github.com/kubataba/memories-creator.git
cd memories-creator
pip install -r requirements.txt
pip install -e .
```

### Additional Requirements

**FFmpeg** (required):  

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

---

### 📁 Итоговая структура проекта для PyPI

```
memories-creator/
├── memories_creator/        
│   ├── __init__.py
│   ├── cli.py
│   ├── main.py
│   ├── config.py
│   ├── music_generator.py
│   ├── video_processor.py
│   ├── transitions.py
│   ├── slide_editor.py
│   └── media_utils.py
│
├── README.md                    
├── LICENSE                       
├── setup.py                     
├── pyproject.toml 
├── config.json             
├── requirements.txt               
├── MANIFEST.in                   
└── .gitignore
```

---   


# 🚀 Quick Start  


### 1. Create Your First Video. 


```bash
# Create configuration
memories config create

# Add your photos to ./photos directory
mkdir photos
cp ~/Pictures/*.jpg photos/

# Generate video
memories create
```  


**Output**: `output/memories_video.mp4`

### 2. With AI Music. 


Edit `config.json`:  

```json
{
  "art_dir": "./photos",
  "output_dir": "./output",
  "generate_music": true,
  "music_prompt": "uplifting piano melody, emotional strings, cinematic",
  "music_model": "small"
}
```  


```bash
memories create
```

### 3. Command Line Options

```bash
# Custom paths
memories create --input ./vacation --output ./videos

# Custom music
memories create --music ./my-song.mp3

# Different transition
memories create --transition ken_burns

# Generate music only
memories music "epic orchestral" --duration 120

# List slides
memories list

# Fix rotations
memories fix 8:180 12:90

# Replace audio
memories audio --file new-music.mp3
```

---

## 📖 Python API

### Basic Usage

```python
from memories_creator import MemoriesCreator, Config

# Load configuration
config = Config.from_file('config.json')

# Create video
creator = MemoriesCreator(config)
creator.create_video()
```

### Generate Music

```python
from memories_creator import MusicGenerator

# Create generator
generator = MusicGenerator(model_name='small')

# Generate soundtrack
music_path = generator.generate(
    prompt="peaceful piano with gentle strings",
    duration=120,
    output_path="output.wav"
)
```

### Batch Processing

```python
from memories_creator import MemoriesCreator, Config

projects = ['vacation', 'wedding', 'family']

for project in projects:
    config = Config()
    config.set('art_dir', f'./{project}')
    config.set('output_dir', f'./output/{project}')
    
    creator = MemoriesCreator(config)
    creator.create_video()
```

---

## 🎛️ Configuration

### Example config.json

```json
{
  "art_dir": "./photos",
  "output_dir": "./output",
  
  "music_prompt": "nostalgic piano, gentle strings",
  "generate_music": true,
  "music_model": "small",
  "music_seamless": true,
  "music_volume": 0.3,
  
  "max_video_size": 1920,
  "fps": 30,
  "seconds_per_photo": 3,
  
  "transition_type": "ken_burns",
  "transition_intensity": 0.1,
  
  "enhance_old_photos": true,
  "year_threshold": 2020,
  
  "video_codec": "hevc_videotoolbox",
  "video_quality": 23
}
```

### Transition Types

- `ken_burns` - Slow zoom + pan (classic)
- `zoom_in` - Smooth zoom in
- `zoom_out` - Smooth zoom out
- `pan` - Horizontal/vertical movement
- `fade_in` - Fade from black
- `fade_out` - Fade to black
- `fade_cross` - Crossfade between images
- `dissolve` - Smooth blend
- `static` - No movement (fast)

### Music Models

- `small` - Fast, ~1.5GB (recommended)
- `medium` - Balanced, ~3.3GB
- `large` - Best quality, ~6.7GB
- `melody` - Melody conditioning

---

## 💻 System Requirements

**Minimum:**  

- Python 3.9+
- 8GB RAM
- macOS 12.3+ or Linux (Ubuntu 20.04+)
- FFmpeg

**Recommended:**  

- Python 3.10+
- 16GB+ RAM
- Apple Silicon (M1/M2/M3/M4) Mac
- macOS 13+
- 20GB disk space (for AI models)

---

## 🐛 Troubleshooting

### MPS not available  

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```
Update macOS to 12.3+ and PyTorch:  
 `pip install --upgrade torch`

### Out of memory  
Reduce `max_video_size` to 1280 and use `transition_type: "static"`

### Music generation fails  

Check disk space (models are 1.5-6GB): `df -h`

### FFmpeg not found  

```bash
# macOS: brew install ffmpeg
# Linux: sudo apt install ffmpeg
```

---

## 📄 License

**CC-BY-NC-4.0** (Creative Commons Attribution-NonCommercial 4.0 International)

This project is licensed for **non-commercial use only**.

You are free to:  

- ✅ Use for personal videos and memories
- ✅ Modify and adapt the code
- ✅ Share with attribution

You must:  

- 📝 Give appropriate credit
- 🔗 Provide a link to the license
- ⚠️ **Not use for commercial purposes**

**Attribution:**  

```
Memories Creator by Eduard Emkuzhev
License: CC-BY-NC-4.0
https://github.com/kubataba/memories-creator
```

Full license: https://creativecommons.org/licenses/by-nc/4.0/

---

## 🙏 Acknowledgments

- **PyTorch** - Deep learning framework
- **MusicGen** - AI music generation (Meta AI)
- **FFmpeg** - Audio/video processing
- **OpenCV** - Video encoding

---

## 📞 Support

- 🐛 [Report Issues](https://github.com/kubataba/memories-creator/issues)
- 💬 [Discussions](https://github.com/kubataba/memories-creator/discussions)
- 📖 [Documentation](https://github.com/kubataba/memories-creator)

---

## 🤝 Contributing

Contributions welcome! This project is licensed under CC-BY-NC-4.0.

```bash
git clone https://github.com/kubataba/memories-creator.git
cd memories-creator
pip install -r requirements.txt
pip install -e .
```

---

**Made with ❤️ to keep your memories alive**
