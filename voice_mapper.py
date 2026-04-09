EMOTION_PROFILES = {
    "happy": {
        "tempo":     +0.30,
        "pitch_st":  +3.0,
        "volume_db": +4.0,
    },
    "angry": {
        "tempo": +0.15,
        "pitch_st": -1.5,
        "volume_db": +6.0,
    },
    "sad": {
        "tempo": -0.30,
        "pitch_st": -3.5,
        "volume_db": -2.0,
    },
    "frustrated": {
        "tempo":     -0.22,
        "pitch_st":  -1.5,
        "volume_db": +2.0,    
    },
    "neutral": {
        "tempo":      0.00,
        "pitch_st":   0.00,
        "volume_db":  0.00,
    },
    "surprise": {
        "tempo":     +0.35,
        "pitch_st":  +4.0,
        "volume_db": +3.0,
    },
    "inquisitive": {
        "tempo":     -0.12,
        "pitch_st":  +1.5,    
        "volume_db":  0.0,
    },
}

def get_voice_params(emotion_result: dict) -> dict:
    emotion   = emotion_result["emotion"]
    intensity = emotion_result["intensity"]
    profile   = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES["neutral"])

    return {
        "emotion":    emotion,
        "intensity":  intensity,
        "tempo":      round(1.0 + profile["tempo"]    * intensity, 3),
        "pitch_st":   round(profile["pitch_st"]       * intensity, 2),
        "volume_db":  round(profile["volume_db"]      * intensity, 1),
        "raw_scores":  emotion_result.get("raw_scores", {}),
        "model_label": emotion_result.get("model_label", ""),
    }