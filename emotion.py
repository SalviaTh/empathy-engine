import re
from transformers import pipeline
_classifier = pipeline(
    task="text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    device=-1
)
#Model Mapping
EMOTION_MAP = {
    "joy":      "happy",
    "anger":    "angry",
    "disgust":  "frustrated",
    "fear":     "inquisitive",
    "sadness":  "sad",
    "surprise": "surprise",
    "neutral":  "neutral",
}

def _punctuation_boost(text: str) -> float:
    """
    Extra intensity from exclamation marks and ALL CAPS words.
    The model doesn't always pick up on typography so we add this manually.
    Capped at 0.25 so it nudges but never dominates.
    """
    exclamations = len(re.findall(r'!', text))
    caps_words   = len(re.findall(r'\b[A-Z]{2,}\b', text))
    return min(0.25, exclamations * 0.07 + caps_words * 0.06)

def detect_emotion(text: str) -> dict:
    results = _classifier(text)[0]

    # Build a clean dict of all scores
    raw_scores = {r["label"]: round(r["score"], 4) for r in results}

    # Pick the highest confidence emotion
    top = max(results, key=lambda r: r["score"])
    model_emotion = top["label"]
    confidence    = top["score"]

    # Map to our voice category
    voice_emotion = EMOTION_MAP.get(model_emotion, "neutral")

    # Final intensity = model confidence + typography boost, capped at 1.0
    intensity = min(1.0, confidence + _punctuation_boost(text))

    return {
        "emotion":    voice_emotion,
        "intensity":  round(intensity, 3),
        "raw_scores": raw_scores,
        "model_label": model_emotion,
    }