import os, uuid
import numpy as np
import librosa
from elevenlabs import ElevenLabs, VoiceSettings
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
from scipy.signal import butter, sosfilt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "static", "audio")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ELEVENLABS_API_KEY =os.environ.get("ELEVENLABS_API_KEY","");
VOICE_ID = "cgSgspJ2msm6clMCkdW9"   
#Voice id: Jessica - Playful, Bright, Warm
MODEL_ID  = "eleven_turbo_v2_5"
#eleven_turbo_v2 because it is fast

client = ElevenLabs(api_key=ELEVENLABS_API_KEY)


def _get_voice_settings(params: dict) -> VoiceSettings:
    emotion   = params["emotion"]
    intensity = params["intensity"]

    if emotion == "happy":
        return VoiceSettings(
            stability=max(0.30, 0.65 - 0.35 * intensity),
            similarity_boost=0.80,
            style=min(0.75, 0.35 + 0.40 * intensity),
            speed=min(1.10, 1.0 + 0.10 * intensity),
        )
    elif emotion == "angry":
        return VoiceSettings(
            stability=max(0.10, 0.45 - 0.35 * intensity),
            similarity_boost=0.70,
            style=min(1.0, 0.55 + 0.45 * intensity),
            speed=min(1.15, 1.0 + 0.15 * intensity),
        )
    elif emotion == "sad":
        return VoiceSettings(
            stability=max(0.55, 0.85 - 0.30 * intensity),
            similarity_boost=0.80,
            style=min(0.50, 0.15 + 0.35 * intensity),
            speed=max(0.72, 1.0 - 0.28 * intensity),
        )

    elif emotion == "frustrated":
        return VoiceSettings(
            stability=max(0.50, 0.80 - 0.30 * intensity), 
            similarity_boost=0.80,
            style=min(0.45, 0.15 + 0.30 * intensity),
            speed=max(0.88, 1.0 - 0.12 * intensity),
        )

    elif emotion == "surprise":
        return VoiceSettings(
            stability=max(0.25, 0.55 - 0.30 * intensity),
            similarity_boost=0.78,
            style=min(0.80, 0.40 + 0.40 * intensity),
            speed=min(1.12, 1.0 + 0.12 * intensity),
        )

    elif emotion == "inquisitive":
        return VoiceSettings(
            stability=max(0.45, 0.70 - 0.25 * intensity),
            similarity_boost=0.82,
            style=min(0.55, 0.20 + 0.35 * intensity),
            speed=max(0.90, 1.0 - 0.10 * intensity),
        )

    else:
        return VoiceSettings(
            stability=0.80,
            similarity_boost=0.85,
            style=0.08,
            speed=1.0,
        )


def _to_numpy(seg: AudioSegment):
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    if seg.channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    samples /= np.iinfo(seg.array_type).max
    return samples, seg.frame_rate

def _to_audiosegment(samples: np.ndarray, sr: int) -> AudioSegment:
    pcm = (samples * 32767).astype(np.int16)
    return AudioSegment(pcm.tobytes(), frame_rate=sr, sample_width=2, channels=1)



def shift_pitch(seg: AudioSegment, semitones: float) -> AudioSegment:
    """Pitch shift without changing duration — librosa phase vocoder."""
    if abs(semitones) < 0.1:
        return seg
    y, sr = _to_numpy(seg)
    shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
    return _to_audiosegment(shifted, sr)

def apply_tremolo(seg: AudioSegment, depth: float = 0.3, freq: float = 4.5) -> AudioSegment:
    """Amplitude wobble — adds vocal tension for frustration."""
    y, sr = _to_numpy(seg)
    t = np.linspace(0, len(y) / sr, len(y))
    modulator = 1.0 - depth * (0.5 + 0.5 * np.sin(2 * np.pi * freq * t))
    return _to_audiosegment(y * modulator, sr)

def boost_high_shelf(seg: AudioSegment, gain_db: float = 4.0, cutoff_hz: float = 3000.0) -> AudioSegment:
    """Brighten voice — adds sparkle/energy for happiness."""
    if abs(gain_db) < 0.5:
        return seg
    y, sr = _to_numpy(seg)
    sos = butter(2, cutoff_hz / (sr / 2), btype='high', output='sos')
    highs = sosfilt(sos, y)
    factor = 10 ** (gain_db / 20)
    boosted = np.clip(y + highs * (factor - 1.0), -1.0, 1.0)
    return _to_audiosegment(boosted, sr)

def add_reverb_tail(seg: AudioSegment, decay: float = 0.25) -> AudioSegment:
    """Subtle reverb bloom — adds wonder/spaciousness for surprise."""
    y, sr = _to_numpy(seg)
    delay_samples = int(sr * 0.06)
    output = y.copy()
    for i in range(delay_samples, len(y)):
        output[i] += decay * output[i - delay_samples]
    return _to_audiosegment(np.clip(output, -1.0, 1.0), sr)

def cut_low_shelf(seg: AudioSegment, cutoff_hz: float = 300.0) -> AudioSegment:
    """Remove low frequencies — makes voice airier for inquisitive tone."""
    y, sr = _to_numpy(seg)
    sos = butter(2, cutoff_hz / (sr / 2), btype='high', output='sos')
    return _to_audiosegment(sosfilt(sos, y), sr)




def _post_process(seg: AudioSegment, params: dict) -> AudioSegment:
    """
    ElevenLabs already handles a lot of the emotion expression.
    We add subtle DSP on top for extra character — pitch and texture.
    These are intentionally lighter than the gTTS version since
    ElevenLabs already sounds expressive.
    """
    emotion   = params["emotion"]
    intensity = params["intensity"]
    pitch_st  = params["pitch_st"]

    if emotion == "happy":
        seg = shift_pitch(seg, pitch_st)
        seg = boost_high_shelf(seg,
                gain_db=1.5 + 2.5 * intensity,
                cutoff_hz=3000)
        seg = seg + (1 + 3 * intensity)
    elif emotion == "angry":

        seg = shift_pitch(seg, pitch_st)
        seg = compress_dynamic_range(seg)
        seg = boost_high_shelf(seg, gain_db=2.0 + 3.0 * intensity, cutoff_hz=2000)
        seg = seg + (3 + 4 * intensity)

    elif emotion == "sad":
        seg = shift_pitch(seg, pitch_st)
        seg = cut_low_shelf(seg, cutoff_hz=80)
        y, sr = _to_numpy(seg)
        sos = butter(2, 4000 / (sr / 2), btype='low', output='sos')
        from scipy.signal import sosfilt
        y = sosfilt(sos, y)
        seg = _to_audiosegment(y, sr)
        seg = seg + (-1 + (-2) * intensity)

    elif emotion == "frustrated":
        seg = shift_pitch(seg, pitch_st)
        seg = apply_tremolo(seg,
                depth=0.08 + 0.18 * intensity, 
                freq=4.5)
        seg = compress_dynamic_range(seg)
        seg = seg + (2 + 3 * intensity)

    elif emotion == "surprise":
        seg = shift_pitch(seg, pitch_st)
        seg = add_reverb_tail(seg, decay=0.12 + 0.18 * intensity)
        seg = boost_high_shelf(seg,
                gain_db=1.0 + 2.5 * intensity,
                cutoff_hz=3200)
        seg = seg + (1 + 2 * intensity)

    elif emotion == "inquisitive":
        seg = shift_pitch(seg, pitch_st)
        seg = cut_low_shelf(seg, cutoff_hz=150 + 100 * intensity)

    seg = normalize(seg)
    silence = AudioSegment.silent(duration=200)
    seg = silence + seg + silence

    return seg


#synthesize function
def synthesize(text: str, params: dict) -> str:
    raw_path   = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}_raw.mp3")
    final_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}_final.mp3")

    voice_settings = _get_voice_settings(params)

    # Wrap in try/except to catch API errors clearly
    try:
        audio_stream = client.text_to_speech.convert(
            voice_id=VOICE_ID,
            text=text,
            model_id=MODEL_ID,
            voice_settings=voice_settings,
            output_format="mp3_44100_128",
        )

        with open(raw_path, "wb") as f:
            for chunk in audio_stream:
                if chunk:
                    f.write(chunk)

    except Exception as e:
        raise RuntimeError(f"ElevenLabs API error: {str(e)}")

    # Load and post-process
    audio = AudioSegment.from_mp3(raw_path)
    audio = audio.set_channels(1).set_frame_rate(44100)
    audio = _post_process(audio, params)
    audio.export(final_path, format="mp3", bitrate="128k")
    os.remove(raw_path)

    return os.path.basename(final_path)