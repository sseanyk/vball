# vball

Turns long recordings of volleyball games into short highlight clips you can post. It uses a custom-trained YOLO model to find rallies, crowd noise and spoken "clip that" calls to pick highlights, and ball tracking to reframe clips to vertical (9:16).

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1. Collect training frames | `extract_frames.py` | Saves one frame every 0.5s from a game video into `training_images/` for labeling. |
| 2. Train the ball detector | `cheat.py` | Downloads the labeled Roboflow dataset and fine-tunes `yolo26s.pt` into `volleyball_yolo.pt`. |
| 3. Cut rallies | `vballCodeV3.py` | You draw a box over the airspace above the net. YOLO tracks the ball, and timestamps with the ball in that box are grouped into rallies and exported as clips. |
| 4. Pick highlights | `save.py`, `highlightReel.py` | Scores each rally by its loudest 1.5s of audio and keeps the above-average ones, or ranks the top N across all games. |
| 4b. Clip on cue | `clipper.py` | Transcribes game audio with faster-whisper and cuts a clip whenever someone says "clip", plus on sustained crowd-noise spikes. Writes a `manifest.csv`. |
| 5. Go vertical | `vertical.py` | Follows the ball with a smoothed 9:16 crop that pans faster when the ball moves faster, then puts the original audio back on. |
| Utility | `lazy.py` | Sorts clips into folders by keywords in their filenames. |

`vballCode.py`, `vballCodeV2.py` and `vballCodeV2.1.py` are earlier versions of the rally detector, starting from motion-based detection.

## Models

- `volleyball_yolo.pt`: the custom volleyball detector trained by `cheat.py`. This is the model the rally, highlight and vertical scripts use.
- `yolo26s.pt`, `yolo26m.pt`: the pretrained Ultralytics YOLO26 base weights it was fine-tuned from.

## Setup

Requires Python 3.11+ and [ffmpeg](https://ffmpeg.org/).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your keys
```

An API key is only needed for `cheat.py` (Roboflow).

The scripts default to `device="mps"` (Apple Silicon GPU). Change it to `"cuda"` or `"cpu"` on other machines. Most scripts set their input and output paths in their `__main__` block, so edit those before running. `clipper.py` takes command-line arguments:

```bash
python clipper.py path/to/games/ --out clips/
```

## License

Ultralytics YOLO is licensed under AGPL-3.0. Code and weights derived from it fall under the same terms.
