# Empathy Engine
### AI-Powered Emotionally Expressive Text-to-Speech

> "The Empathy Engine, dynamically modulates the vocal characteristics of synthesized speech based on the detected emotion of the source text bridging gap between text-90'sed sentiment and expressive, human-like audio output, moving beyond monotonic delivery to achieve emotional resonance.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Setup & Installation](#setup--installation)
- [Using the Web Interface](#using-the-web-interface)
- [Design Choices](#design-choices)
- [Emotion-to-Voice Mapping Logic](#emotion-to-voice-mapping-logic)
- [Tech Stack](#tech-stack)

## How It Works

```
User types text
      ↓
HuggingFace transformer model
detects emotion + confidence score
      ↓
Confidence score becomes intensity (0.0 – 1.0)
      ↓
Intensity scales vocal parameters
(tempo, pitch semitones, volume dB)
      ↓
ElevenLabs API generates neural speech
with emotion-matched voice settings
      ↓
librosa + pydub apply DSP effects
(pitch shift, tremolo, EQ, reverb)
      ↓
Final .mp3 served to browser
```

**Supported emotions:** Happy · Angry · Sad · Frustrated· Surprise · Inquisitive · Neutral

---

## Setup & Installation

### Prerequisites

- Python 3.10 or higher
- ffmpeg installed on your system
- An ElevenLabs account (free tier works)

### 1. Install ffmpeg

**Windows:**
```
1. Download from https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
2. Unzip and rename folder to "ffmpeg"
3. Move to C:\ffmpeg
4. Add C:\ffmpeg\bin to your system PATH
5. Open a new terminal and verify: ffmpeg -version
```

**Mac:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

### 2. Clone the repository

```bash
git clone https://github.com/SalviaTh/empathy-engine.git
cd empathy-engine
```

### 3. Create a virtual environment

```bash
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> `torch` is a large package (~2GB). This step may take several minutes.

```
```
### 5. Get your ElevenLabs API key

```
1. Go to https://elevenlabs.io and sign up (free)
2. Go to Profile → API Key
3. Copy your key — looks like: sk_xxxxxxxxxxxxxxxxxxxxxxxx
```

### 6. Set your API key as an environment variable

create a `.env` file in the project root:
```
ELEVENLABS_API_KEY=sk_xxxxxxxxxxxxxxxxxxxxxxxx
```

### 7. Pre-download the HuggingFace model

This downloads the emotion model (~330MB) once and caches it locally:

```bash
python preload_model.py
```

---

## Running the App

```bash
python app.py
```

You should see:
```
Device set to use cpu
* Running on http://127.0.0.1:5000
* Debugger is active!
```

Open your browser and go to: **http://127.0.0.1:5000**

---

## Using the Web Interface

1. Type any sentence into the text area
2. Click **▶ Speak it**
3. Wait 3–5 seconds for processing
4. The detected emotion badge and audio player appear
5. Click **⬇ Download audio** to save the .mp3

**Test sentences to hear the difference:**

| Emotion | Example |
|  Happy | `This is the best news I've heard all year!` |
|  Angry | `This is completely unacceptable and I am furious!` |
|  Sad | `I can't believe they're gone. I really miss them.` |
|  Frustrated|`That is absolutely revolting, I can't even look at it.` |
|  Surprise | `Wait you actually won? I had no idea!` |
|  Inquisitive | `I wonder why that keeps happening every single time?` |
|  Neutral | `The meeting is scheduled for three o'clock tomorrow.` |

---

## Design Choices

### Why HuggingFace over VADER?

The first version used VADER (Valence Aware Dictionary and sEntiment Reasoner), a rule-based tool that only detects positive/negative/neutral polarity from a word list. It fails on sentences like *"I can't believe how good this is"*.


So I use `j-hartmann/emotion-english-distilroberta-base`, a transformer model trained on millions of real-world sentences. It outputs 7 distinct emotion classes with confidence scores, and understands context rather than just keywords. The confidence score (0.0–1.0) becomes the **intensity** value that scales all vocal parameters meaning the same emotion sounds subtly different at low confidence vs high confidence.

### Why ElevenLabs over gTTS?

gTTS (Google Translate TTS) generates audio and works well in tts translation but sounds completely robotic.

ElevenLabs uses a deep learning vocoder and exposes direct controls: `stability` , `style` (emotional intensity), and `speed`. This means emotion is baked into the voice generation itself, not bolted on afterward. The post-processing DSP layer then adds fine-grained texture on top.The audio doesn't sound robotic.

### Why librosa + pydub together?

They solve different problems:

- **pydub** handles file I/O (loading mp3, exporting, volume in dB) but its speed change alters pitch simultaneously — like playing a tape faster, which sounds unnatural
- **librosa** uses a phase vocoder algorithm to change pitch and tempo independently — pitch shift without speed change, and speed change without pitch shift

Both are needed to have full, independent control over all vocal dimensions.

### Why post-process if ElevenLabs already handles emotion?

ElevenLabs controls the voice character and expressiveness at generation time. The DSP layer adds acoustic texture that ElevenLabs cannot do natively:

- **Tremolo** (amplitude wobble at 4.5Hz) makes angry voices sound physically tense
- **High shelf EQ boost** adds brightness and presence to happy speech
- **Reverb tail** adds a bloom of spaciousness that makes surprise sound wonderstruck
- **Low shelf cut** removes chest resonance making inquisitive speech sound airy and hesitant
- **Low-pass filter** on sad speech muffles the highs making it sound heavy and withdrawn

---

## Emotion-to-Voice Mapping Logic

The core mapping philosophy is: **each emotion has a physical vocal reality**, and we try to replicate that reality through parameters.

### Intensity scaling

Every parameter is multiplied by `intensity` (the model's confidence score, 0.0–1.0):

```
final_tempo = 1.0 + (base_tempo_change × intensity)
```

So *"This is good"* (joy at 0.42 confidence) gets a subtle pitch rise, while *"THIS IS THE BEST DAY EVER!!!"* (joy at 0.97) gets the full treatment. Punctuation and ALL CAPS words add a small boost on top.

### Parameter table

| Emotion | Tempo | Pitch | Volume | ElevenLabs Stability | Style | DSP Effect |
|---|---|---|---|---|---|---|
| Happy | +30% faster | +3.0st higher | +4dB | Low (expressive) | High | High shelf EQ boost |
| Angry | +15% faster | -1.5st lower | +6dB | Very low (unstable) | Very high | Compress + presence boost |
| Sad | -30% slower | -3.0st lower | -2dB | High (controlled) | Low | Low-pass filter (muffled) |
| Frustrated| -10% slower | -1.0st lower | +2dB | Medium | Low-medium | Low shelf cut (nasal) |
| Surprise | +35% faster | +4.0st higher | +3dB | Low (expressive) | High | Reverb bloom |
| Inquisitive | -12% slower | +1.5st higher | 0dB | Medium-high | Medium | Low shelf cut (airy) |
| Neutral | No change | No change | 0dB | High (stable) | Minimal | None |

### Why these specific mappings?

**Happy → fast + high pitch + bright EQ**
Excited speech naturally speeds up. Pitch rises with positive arousal. Boosting 3kHz+ adds the "smile" you can hear in a genuinely happy voice.

**Angry → fast + low pitch + loud + compressed**
Anger is high energy but controlled — sharp and clipped, not chipmunk-fast. Low pitch adds weight and authority. Compression flattens dynamics making it sound strained. For customer service context, stability is kept higher than pure anger to maintain a professional edge.

**Sad → slow + very low pitch + quiet + muffled highs**
Sadness is physically deflating — slower breathing, lower register, less projection. The low-pass filter removes high-frequency clarity making the voice sound like it's coming through grief. Quieter volume reflects withdrawn energy.

**Frustrated slightly slow + flat + nasal EQ**
Frustrated is dismissive rather than energetic. Low expressiveness (high stability) keeps it dry and flat. Cutting low frequencies removes warmth and chest resonance, pushing the voice into a nasal, clipped register.

**Surprise → fastest + highest pitch + reverb**
Surprise triggers a sharp intake of breath and a jump in pitch. The reverb tail adds an acoustic "bloom" — the feeling of a space opening up around the voice, matching the psychological sensation of surprise expanding your attention.

**Inquisitive → slightly slow + slight pitch rise + airy EQ**
Questions naturally end with a rising inflection. Slowing slightly adds thoughtfulness. Cutting the low shelf removes grounded chest resonance making the voice sound open and wondering rather than declarative.

---


## Tech Stack

| Component | Technology | Purpose |

| Web framework | Flask | HTTP server + routing |
| Emotion detection | HuggingFace Transformers (`j-hartmann/emotion-english-distilroberta-base`) | 7-class emotion classification |
| Text-to-speech | ElevenLabs API (`eleven_flash_v2`) | Neural voice generation |
| Pitch shifting | librosa (phase vocoder) | Independent pitch control |
| Audio editing | pydub | Loading, volume, export |
| DSP effects | scipy (Butterworth filters) | EQ shaping per emotion |
| Deployment | Render | Free cloud hosting |

---