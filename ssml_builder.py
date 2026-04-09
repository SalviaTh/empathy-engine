# SSML is an XML dialect that tells TTS engines HOW to speak — not just WHAT.

def build_ssml(text: str, params: dict) -> str:
    emotion   = params.get("emotion", "neutral")
    tempo     = params.get("tempo", 1.0)
    pitch_st  = params.get("pitch_st", 0.0)
    volume_db = params.get("volume_db", 0.0)

    # Convert our float values back to SSML string format
    rate_pct   = f"{int((tempo - 1.0) * 100):+d}%"
    pitch_ssml = f"{pitch_st:+.1f}st"
    vol_ssml   = f"{volume_db:+.1f}dB"

    # Emotion-specific inner markup
    if emotion == "surprise":
        inner = f'<emphasis level="strong">{text}</emphasis>'
    elif emotion == "frustrated":
        inner = f'<emphasis level="moderate">{text}</emphasis>'
    elif emotion == "inquisitive":
        # Add a short breath pause before sentences that end in ?
        inner = text.replace("?", '<break time="150ms"/>?')
    else:
        inner = text

    return f"""<speak>
  <prosody rate="{rate_pct}" pitch="{pitch_ssml}" volume="{vol_ssml}">
    {inner}
  </prosody>
</speak>"""