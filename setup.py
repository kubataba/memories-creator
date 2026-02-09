#!/usr/bin/env python3
"""
Memories Creator - Setup Script
Professional video slideshow creation with Apple-style effects
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="memories-creator",
    version="1.0.0",
    author="Memories Creator Team",
    author_email="your.email@example.com",  # Replace with your email
    description="Professional video slideshow creation with Apple-style transitions and AI-generated music",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kubataba/memories-creator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Video :: Conversion",
        "License :: Other/Proprietary License", 
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Natural Language :: English",
    ],
    license="CC-BY-NC-4.0",
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "tqdm>=4.65.0",
        "soundfile>=0.12.1",
        "librosa>=0.10.0",
        "scipy>=1.11.0",
        "txtai>=7.0.0",
        "transformers>=4.35.0",
        "resampy>=0.4.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "memories=memories_creator.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "memories_creator": ["config_template.json"],
    },
    project_urls={
        "Bug Reports": "https://github.com/kubataba/memories-creator/issues",
        "Source": "https://github.com/kubataba/memories-creator",
        "Documentation": "https://github.com/kubataba/memories-creator",
    },
    keywords=[
        "video",
        "slideshow",
        "photos",
        "ai-music",
        "musicgen",
        "apple-silicon",
        "memories",
        "video-editing",
        "photo-slideshow",
        "transitions",
    ],
)