"""
Music Generator Module for Memories Creator
Core functionality for AI music generation using Facebook's MusicGen models
via txtai TextToAudio pipeline
"""

import os
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple, Union, List
import warnings
import time

# Primary import for music generation - uses txtai's TextToAudio wrapper
try:
    from txtai.pipeline import TextToAudio
    MUSIC_GENERATION_AVAILABLE = True
except ImportError:
    MUSIC_GENERATION_AVAILABLE = False
    warnings.warn(
        "Music generation dependencies not installed. "
        "Install with: pip install txtai[audio]"
    )

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class MusicGenerator:
    """
    Main class for AI music generation and audio processing
    
    Features:
    - Text-to-music generation using pre-trained MusicGen models
    - Audio loading, processing, and format conversion
    - Seamless music extension through intelligent looping
    - Fallback generation for reliability
    - Video synchronization support
    """
    
    # Pre-trained MusicGen model variants from Facebook Research
    AVAILABLE_MODELS = {
        'small': 'facebook/musicgen-stereo-small',     # Fastest, 300M parameters
        'medium': 'facebook/musicgen-stereo-medium',   # Balanced quality, 1.5B parameters
        'large': 'facebook/musicgen-stereo-large',     # Best quality, 3.3B parameters
        'melody': 'facebook/musicgen-stereo-melody',   # Supports melody conditioning
    }
    
    def __init__(self, config=None, model_name: str = 'small'):
        """
        Initialize the music generator
        
        Args:
            config: Application configuration dictionary
            model_name: Name of the MusicGen model to use
        """
        self.config = config
        # Use model from config if provided, otherwise default to 'small'
        self.model_name = model_name or (config.get('music_model', 'small') if config else 'small')
        
        # Core generator instance (txtai TextToAudio wrapper)
        self.generator = None
        self._model_loaded = False
        
        # Resolve model path from model name
        if self.model_name in self.AVAILABLE_MODELS:
            self.model_path = self.AVAILABLE_MODELS[self.model_name]
        else:
            self.model_path = self.model_name  # Allow custom model paths
        
        # Load settings from configuration
        if config:
            self.default_seamless = config.get('music_seamless', True)
        else:
            self.default_seamless = True
        
        # Statistics tracking
        self.statistics = {
            'total_generations': 0,
            'successful_generations': 0,
            'fallback_used': 0,
            'total_duration_generated': 0.0
        }
    
    def _ensure_model_loaded(self):
        """
        Lazy loading of the MusicGen model
        
        Loads the model only when first needed to save memory and startup time
        """
        if self._model_loaded:
            return
        
        if not MUSIC_GENERATION_AVAILABLE:
            raise ImportError(
                "Music generation requires txtai[audio]. "
                "Install with: pip install txtai[audio]"
            )
        
        print(f"🎵 Loading MusicGen model: {self.model_path}")
        print("   (First load may take 1-2 minutes to download model weights)")
        
        try:
            # Initialize txtai's TextToAudio with the MusicGen model
            self.generator = TextToAudio(self.model_path)
            self._model_loaded = True
            print("✅ Music model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load music model: {e}")
            raise RuntimeError(f"Music model loading failed: {e}")
    
    def generate(
        self,
        prompt: str = None,
        duration: float = None,
        output_path: Optional[str] = None,
        seamless: Optional[bool] = None
    ) -> str:
        """
        Generate music from a text description
        
        Core generation workflow:
        1. Load model if not already loaded
        2. Generate initial audio segment using MusicGen
        3. Extend/loop audio to reach target duration
        4. Apply post-processing (normalization, fades)
        5. Save to output file
        
        Args:
            prompt: Text description of desired music style
            duration: Target duration in seconds
            output_path: Custom output file path (optional)
            seamless: Enable seamless looping for longer durations
            
        Returns:
            Path to the generated audio file
            
        Raises:
            RuntimeError: If generation fails completely
        """
        self.statistics['total_generations'] += 1
        
        # Resolve parameters (use config defaults if not provided)
        prompt = self._resolve_parameter('prompt', prompt, 'uplifting piano melody')
        duration = self._resolve_parameter('duration', duration, 30.0)
        seamless = self._resolve_parameter('seamless', seamless, self.default_seamless)
        
        print(f"🎼 Generating music: '{prompt}'")
        print(f"   Target duration: {duration:.1f}s | Model: {self.model_name}")
        
        # Ensure model is loaded before generation
        self._ensure_model_loaded()
        
        try:
            # Generate audio using MusicGen
            audio_data, sample_rate = self._generate_audio(prompt, duration, seamless)
            
            # Apply final processing and save
            output_path = self._save_audio_file(audio_data, sample_rate, prompt, output_path)
            
            # Update success statistics
            self.statistics['successful_generations'] += 1
            self.statistics['total_duration_generated'] += duration
            
            return output_path
            
        except Exception as e:
            print(f"❌ Music generation failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Attempt fallback generation
            print("🔄 Attempting fallback audio generation...")
            try:
                fallback_path = self._create_fallback_audio(prompt, duration, output_path)
                self.statistics['fallback_used'] += 1
                return fallback_path
            except Exception as fallback_error:
                print(f"❌ Fallback generation also failed: {fallback_error}")
                raise RuntimeError(f"All music generation attempts failed")
    
    def _resolve_parameter(self, param_name: str, value, default):
        """
        Resolve parameter value with configuration fallback
        
        Priority order:
        1. Directly provided value
        2. Configuration setting
        3. Default value
        """
        if value is not None:
            return value
        elif self.config:
            return self.config.get(f'music_{param_name}', default)
        else:
            return default
    
    def _generate_audio(
        self,
        prompt: str,
        target_duration: float,
        seamless: bool
    ) -> Tuple[np.ndarray, int]:
        """
        Core audio generation logic
        
        Generates initial audio segment and extends it to target duration
        using intelligent looping techniques.
        """
        print("   Generating initial audio segment...")
        
        # Generate initial audio using MusicGen
        # Note: txtai's TextToAudio doesn't support duration parameter,
        # so we generate whatever length it produces and extend it
        result = self.generator(prompt)
        
        if not result or not isinstance(result, tuple) or len(result) < 2:
            raise ValueError("Invalid response from music generator")
        
        raw_audio, sample_rate = result[0], result[1]
        
        # Convert to numpy array for processing
        if not isinstance(raw_audio, np.ndarray):
            raw_audio = np.array(raw_audio)
        
        # Normalize audio array shape to (channels, samples)
        audio_data = self._normalize_audio_shape(raw_audio)
        
        # Calculate generated duration
        generated_duration = audio_data.shape[-1] / sample_rate
        print(f"   Initial segment: {generated_duration:.1f}s @ {sample_rate}Hz")
        
        # Extend audio to reach target duration if needed
        if generated_duration < target_duration:
            audio_data = self._extend_audio_to_duration(
                audio_data, sample_rate, target_duration, seamless
            )
        
        # Apply audio processing (normalization, fades)
        audio_data = self._process_audio(audio_data, sample_rate)
        
        return audio_data, sample_rate
    
    def _normalize_audio_shape(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio array to standard shape (channels, samples)
        
        Handles various input shapes:
        - 1D arrays (mono) -> (1, samples)
        - 2D arrays with channels first -> (channels, samples)
        - 2D arrays with samples first -> transpose to (channels, samples)
        """
        if len(audio.shape) == 1:
            # Mono audio: convert to 2D with single channel
            return audio[np.newaxis, :]
        elif len(audio.shape) == 2:
            if audio.shape[0] == 1 or audio.shape[0] == 2:
                # Already correct shape: (channels, samples)
                return audio
            elif audio.shape[1] == 1 or audio.shape[1] == 2:
                # Transpose: (samples, channels) -> (channels, samples)
                return audio.T
            else:
                # Unknown 2D shape, average to mono
                mono = np.mean(audio, axis=1, keepdims=True)
                return mono.T
        else:
            # Higher dimensional, flatten to mono
            return audio.flatten()[np.newaxis, :]
    
    def _extend_audio_to_duration(
        self,
        audio: np.ndarray,
        sample_rate: int,
        target_duration: float,
        seamless: bool
    ) -> np.ndarray:
        """
        Extend audio to reach target duration using intelligent looping
        
        Strategies:
        - For very short audio (< 2s): pad with silence
        - For longer audio: loop with crossfade transitions
        - Seamless mode uses smooth crossfades between loops
        """
        target_samples = int(target_duration * sample_rate)
        current_samples = audio.shape[-1]
        
        # Calculate how many loops needed
        loops_needed = int(np.ceil(target_samples / current_samples))
        
        if loops_needed <= 1:
            # No looping needed, just pad if slightly short
            if current_samples < target_samples:
                padding = np.zeros((audio.shape[0], target_samples - current_samples), 
                                 dtype=audio.dtype)
                return np.concatenate([audio, padding], axis=1)
            return audio
        
        print(f"   Extending audio: {loops_needed} loops needed")
        
        if seamless and current_samples > sample_rate * 2:  # At least 2 seconds
            # Use seamless looping with crossfades
            return self._seamless_loop(audio, target_samples, sample_rate)
        else:
            # Simple repetition
            repeated = np.tile(audio, loops_needed)
            return repeated[:, :target_samples]
    
    def _seamless_loop(
        self,
        audio: np.ndarray,
        target_samples: int,
        sample_rate: int
    ) -> np.ndarray:
        """
        Create seamless loops with crossfade transitions
        
        Uses raised cosine crossfade envelope for smooth transitions
        between loop repetitions.
        """
        crossfade_duration = 0.5  # 500ms crossfade
        crossfade_samples = int(sample_rate * crossfade_duration)
        
        # Ensure crossfade is not too long relative to audio
        if crossfade_samples > audio.shape[-1] // 4:
            crossfade_samples = audio.shape[-1] // 4
        
        loops_needed = int(np.ceil(target_samples / audio.shape[-1]))
        
        # Create crossfade envelopes
        if crossfade_samples > 0:
            t = np.linspace(0, np.pi, crossfade_samples)
            fade_out = (np.cos(t) + 1) / 2  # Raised cosine fade out
            fade_in = 1 - fade_out          # Complementary fade in
        else:
            fade_out = fade_in = np.array([])
        
        looped_segments = []
        
        for i in range(loops_needed):
            segment = audio.copy()
            
            if i > 0 and len(looped_segments) > 0 and crossfade_samples > 0:
                # Apply crossfade between segments
                previous_segment = looped_segments[-1]
                
                for channel in range(audio.shape[0]):
                    # Ensure segments are long enough for crossfade
                    if (previous_segment.shape[-1] >= crossfade_samples and 
                        segment.shape[-1] >= crossfade_samples):
                        
                        # Fade out end of previous segment
                        previous_segment[channel, -crossfade_samples:] *= fade_out
                        # Fade in start of current segment
                        segment[channel, :crossfade_samples] *= fade_in
                        # Mix overlapping sections
                        previous_segment[channel, -crossfade_samples:] += segment[channel, :crossfade_samples]
                
                # Add remaining part of current segment
                if segment.shape[-1] > crossfade_samples:
                    looped_segments.append(segment[:, crossfade_samples:])
            else:
                looped_segments.append(segment)
        
        # Concatenate all segments and trim to target length
        result = np.concatenate(looped_segments, axis=1)
        return result[:, :target_samples]
    
    def _process_audio(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        Apply audio post-processing
        
        Includes:
        - Peak normalization
        - Fade in/out application
        - Quality validation
        """
        # Peak normalization to prevent clipping
        peak_value = np.max(np.abs(audio))
        if peak_value > 0:
            audio = audio / peak_value * 0.8  # Normalize to -3dB headroom
        
        # Apply fade in/out from config
        fade_in = self.config.get('music_fade_in', 2.0) if self.config else 2.0
        fade_out = self.config.get('music_fade_out', 3.0) if self.config else 3.0
        
        audio = self._apply_fades(audio, sample_rate, fade_in, fade_out)
        
        return audio
    
    def _apply_fades(
        self,
        audio: np.ndarray,
        sample_rate: int,
        fade_in: float,
        fade_out: float
    ) -> np.ndarray:
        """
        Apply fade in and fade out envelopes to audio
        
        Uses quadratic fade curves for smooth transitions
        """
        total_samples = audio.shape[-1]
        
        # Calculate fade samples
        fade_in_samples = int(sample_rate * fade_in)
        fade_out_samples = int(sample_rate * fade_out)
        
        # Adjust fades if audio is too short
        fade_in_samples = min(fade_in_samples, total_samples // 4)
        fade_out_samples = min(fade_out_samples, total_samples // 4)
        
        # Apply fade in (quadratic curve)
        if fade_in_samples > 0:
            fade_in_curve = np.linspace(0, 1, fade_in_samples) ** 2
            for channel in range(audio.shape[0]):
                audio[channel, :fade_in_samples] *= fade_in_curve
        
        # Apply fade out (quadratic curve)
        if fade_out_samples > 0:
            fade_out_curve = np.linspace(1, 0, fade_out_samples) ** 2
            for channel in range(audio.shape[0]):
                audio[channel, -fade_out_samples:] *= fade_out_curve
        
        return audio
    
    def _save_audio_file(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: str,
        output_path: Optional[str]
    ) -> str:
        """
        Save audio data to disk
        
        Handles both mono and stereo formats, creates necessary directories,
        and provides detailed output information.
        """
        # Generate output path if not provided
        if output_path is None:
            output_path = self._generate_output_path(prompt)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        
        # Save audio in appropriate format
        if audio.shape[0] == 2:
            # Stereo audio: save as (samples, 2) for soundfile
            sf.write(output_path, audio.T, sample_rate)
        else:
            # Mono audio
            sf.write(output_path, audio[0], sample_rate)
        
        # Calculate and display file information
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        duration_seconds = audio.shape[-1] / sample_rate
        
        print(f"✅ Music saved successfully")
        print(f"   File: {output_path}")
        print(f"   Duration: {duration_seconds:.1f}s | Size: {file_size_mb:.1f} MB")
        
        return output_path
    
    def _generate_output_path(self, prompt: str) -> str:
        """
        Generate a unique output filename based on prompt and timestamp
        
        Creates clean, readable filenames that include:
        - Sanitized prompt text
        - Timestamp for uniqueness
        - Organized directory structure
        """
        # Sanitize prompt for filename (alphanumeric and underscores only)
        sanitized_prompt = "".join(
            c if c.isalnum() else "_" for c in prompt[:40]
        )
        
        # Determine output directory from config or use default
        if self.config and self.config.get('output_dir'):
            base_dir = Path(self.config.get('output_dir'))
        else:
            base_dir = Path("./output")
        
        # Create music subdirectory
        music_dir = base_dir / "music"
        music_dir.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp for uniqueness
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Construct final path
        filename = f"music_{sanitized_prompt}_{timestamp}.wav"
        return str(music_dir / filename)
    
    def _create_fallback_audio(
        self,
        prompt: str,
        duration: float,
        output_path: Optional[str]
    ) -> str:
        """
        Generate synthetic fallback audio when AI generation fails
        
        Creates simple, pleasant synthetic music as a reliable fallback.
        Useful for:
        - Network connectivity issues
        - Model loading failures
        - Generation errors
        """
        print("🎹 Generating synthetic fallback audio...")
        
        sample_rate = 32000
        total_samples = int(duration * sample_rate)
        time_array = np.linspace(0, duration, total_samples)
        
        # Create a pleasant A minor chord progression
        audio_waveform = np.zeros(total_samples)
        chord_frequencies = [220.0, 277.0, 330.0]  # A minor chord frequencies
        
        for i, frequency in enumerate(chord_frequencies):
            # Decreasing amplitude for higher harmonics
            amplitude = 0.3 / (i + 1)
            # Generate sine wave with slight detuning for richness
            detuning = 1.0 + (np.random.random() * 0.02 - 0.01)  # ±1% detune
            sine_wave = amplitude * np.sin(2 * np.pi * frequency * detuning * time_array)
            
            # Add gentle tremolo effect
            tremolo_depth = 0.1
            tremolo_rate = 5.0  # Hz
            tremolo = 1.0 - tremolo_depth * np.sin(2 * np.pi * tremolo_rate * time_array)
            sine_wave *= tremolo
            
            audio_waveform += sine_wave
        
        # Convert to stereo
        audio_waveform = audio_waveform[np.newaxis, :]  # Shape: (1, samples)
        stereo_audio = np.tile(audio_waveform, (2, 1))   # Shape: (2, samples)
        
        # Normalize to prevent clipping
        peak_value = np.max(np.abs(stereo_audio))
        if peak_value > 0:
            stereo_audio = stereo_audio / peak_value * 0.8
        
        # Apply fades
        stereo_audio = self._apply_fades(stereo_audio, sample_rate, 2.0, 3.0)
        
        # Generate output path if not provided
        if output_path is None:
            output_path = self._generate_output_path(prompt + "_fallback")
        
        # Save fallback audio
        sf.write(output_path, stereo_audio.T, sample_rate)
        
        print(f"✅ Fallback audio created: {output_path}")
        return output_path
    
    def load_audio(
        self,
        file_path: str,
        target_duration: Optional[float] = None,
        target_sample_rate: int = 44100,
        mono: bool = False
    ) -> Tuple[np.ndarray, int]:
        """
        Load and process an existing audio file
        
        Handles:
        - File existence validation
        - Format conversion (mono/stereo)
        - Sample rate conversion
        - Duration adjustment
        
        Args:
            file_path: Path to audio file
            target_duration: Desired duration (None = keep original)
            target_sample_rate: Target sample rate for output
            mono: Convert to mono if True
            
        Returns:
            Processed audio data and sample rate
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        print(f"🎵 Loading audio file: {file_path}")
        
        # Load audio using soundfile
        audio_data, original_sample_rate = sf.read(file_path)
        
        # Normalize audio shape
        audio_data = self._normalize_audio_shape(audio_data)
        
        # Convert to mono if requested
        if mono and audio_data.shape[0] > 1:
            audio_data = np.mean(audio_data, axis=0, keepdims=True)
        
        # Resample to target sample rate if needed
        if original_sample_rate != target_sample_rate:
            audio_data = self._resample_audio(audio_data, original_sample_rate, target_sample_rate)
            sample_rate = target_sample_rate
        else:
            sample_rate = original_sample_rate
        
        # Adjust duration if specified
        if target_duration is not None:
            target_samples = int(target_duration * sample_rate)
            current_samples = audio_data.shape[-1]
            
            if current_samples < target_samples:
                # Extend audio by looping
                audio_data = self._extend_audio_to_duration(
                    audio_data, sample_rate, target_duration, seamless=True
                )
            elif current_samples > target_samples:
                # Trim to target duration
                audio_data = audio_data[:, :target_samples]
        
        return audio_data, sample_rate
    
    def _resample_audio(
        self,
        audio: np.ndarray,
        original_rate: int,
        target_rate: int
    ) -> np.ndarray:
        """
        Resample audio to target sample rate
        
        Uses librosa if available, falls back to linear interpolation
        """
        if original_rate == target_rate:
            return audio
        
        num_channels = audio.shape[0]
        resampled_channels = []
        
        for channel_idx in range(num_channels):
            channel_data = audio[channel_idx]
            
            if LIBROSA_AVAILABLE:
                # High-quality resampling with librosa
                resampled = librosa.resample(
                    channel_data,
                    orig_sr=original_rate,
                    target_sr=target_rate
                )
            else:
                # Fallback to linear interpolation
                original_length = len(channel_data)
                target_length = int(original_length * target_rate / original_rate)
                sample_positions = np.linspace(0, original_length - 1, target_length)
                resampled = np.interp(sample_positions, np.arange(original_length), channel_data)
            
            resampled_channels.append(resampled)
        
        return np.stack(resampled_channels)
    
    def prepare_for_video(
        self,
        audio_source: Union[str, Tuple[np.ndarray, int]],
        video_duration: float,
        volume: float = 0.3,
        fade_in: float = 2.0,
        fade_out: float = 3.0,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Prepare audio for video synchronization
        
        Processes audio to match video requirements:
        - Adjusts duration to video length
        - Applies volume normalization
        - Adds fade in/out transitions
        - Saves in appropriate format
        
        Args:
            audio_source: Audio file path or (audio_data, sample_rate) tuple
            video_duration: Target video duration in seconds
            volume: Volume multiplier (0.0 to 1.0)
            fade_in: Fade in duration in seconds
            fade_out: Fade out duration in seconds
            output_path: Custom output path (optional)
            
        Returns:
            Path to processed audio file
        """
        # Use configuration values if available
        if self.config:
            volume = self.config.get('music_volume', volume)
            fade_in = self.config.get('music_fade_in', fade_in)
            fade_out = self.config.get('music_fade_out', fade_out)
        
        # Load audio if file path provided
        if isinstance(audio_source, str):
            audio_data, sample_rate = self.load_audio(
                audio_source,
                target_duration=video_duration
            )
        else:
            audio_data, sample_rate = audio_source
        
        # Apply volume adjustment
        audio_data = audio_data * volume
        
        # Apply fade transitions
        audio_data = self._apply_fades(audio_data, sample_rate, fade_in, fade_out)
        
        # Determine output path
        if output_path is None:
            output_path = "processed_music.wav"
        
        # Save processed audio
        if audio_data.shape[0] == 2:
            sf.write(output_path, audio_data.T, sample_rate)
        else:
            sf.write(output_path, audio_data[0], sample_rate)
        
        print(f"✅ Audio prepared for video: {output_path}")
        return output_path
    
    def print_statistics(self):
        """Display music generation statistics"""
        print("\n📊 Music Generation Statistics:")
        print(f"   Total generations attempted: {self.statistics['total_generations']}")
        print(f"   Successful generations: {self.statistics['successful_generations']}")
        print(f"   Fallback audio used: {self.statistics['fallback_used']}")
        print(f"   Total duration generated: {self.statistics['total_duration_generated']:.1f}s")
        
        if self.statistics['total_generations'] > 0:
            success_rate = (
                self.statistics['successful_generations'] / 
                self.statistics['total_generations'] * 100
            )
            print(f"   Success rate: {success_rate:.1f}%")
    
    def get_available_models(self) -> List[str]:
        """Return list of available model names"""
        return list(self.AVAILABLE_MODELS.keys())


# Module-level convenience functions for easy access
def generate_music(
    prompt: str = None,
    duration: float = None,
    output_path: Optional[str] = None,
    model: str = 'small',
    seamless: bool = True,
    config = None
) -> str:
    """
    Quick access function for music generation
    
    Args:
        prompt: Music description
        duration: Target duration in seconds
        output_path: Output file path
        model: MusicGen model name
        seamless: Enable seamless looping
        config: Configuration dictionary
        
    Returns:
        Path to generated audio file
    """
    generator = MusicGenerator(config=config, model_name=model)
    return generator.generate(prompt, duration, output_path, seamless)


def load_music(
    file_path: str,
    duration: Optional[float] = None
) -> Tuple[np.ndarray, int]:
    """
    Quick access function for audio loading
    
    Args:
        file_path: Path to audio file
        duration: Target duration (optional)
        
    Returns:
        Audio data and sample rate
    """
    generator = MusicGenerator()
    return generator.load_audio(file_path, target_duration=duration)