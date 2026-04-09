
from flask import Flask, request, jsonify, render_template
from emotion import detect_emotion
from voice_mapper import get_voice_params
from ssml_builder import build_ssml
from tts_engine import synthesize

app = Flask(__name__)
@app.route("/")              
def index():
    return render_template("index.html")

@app.route("/synthesize", methods=["POST"])
def synthesize_route():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        emotion_result = detect_emotion(text)
        params         = get_voice_params(emotion_result)
        ssml           = build_ssml(text, params)
        filename       = synthesize(text, params)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "audio_url":   f"/static/audio/{filename}",
        "emotion":     params["emotion"],
        "model_label": params["model_label"],
        "intensity":   params["intensity"],
        "tempo":       params["tempo"],
        "pitch_st":    params["pitch_st"],
        "volume_db":   params["volume_db"],
        "raw_scores":  params["raw_scores"],
        "ssml":        ssml,
    })

if __name__ == "__main__":
    app.run(debug=True)