# Smart Parking Occupancy Monitor

A live parking-lot occupancy monitor: point it at a parking lot and it
reports, bay by bay, which spaces are **FREE** and which are **OCCUPIED** —
with running totals and an occupancy percentage drawn on screen. This is
the core of the "find a free space" guidance systems used in malls, offices
and airports.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-real--mode-orange)

## How occupancy works

1. **Detect** — every frame produces vehicle boxes through one detector
   interface: YOLOv8 in real mode, the simulation scene in demo mode.
2. **Assign** — each parking bay is a polygon. A bay is **OCCUPIED** when
   the *centre point of a detected vehicle box falls inside the bay
   polygon*, and **FREE** when no vehicle centre is inside it.
3. **Stabilise** — a bay only changes state after several consecutive
   frames agree (default 4), so a jittery box on a bay edge can't make the
   display flicker.
4. **Report** — bays are outlined green (FREE) or red (OCCUPIED) with
   their ID and state, and a HUD shows total bays, free, occupied,
   occupancy % and FPS.

Bays come from an even grid (`--rows` / `--cols`, default 2×6 = 12 bays)
laid over the lot region, or from a JSON file (`--spots`) with custom
polygons for a real camera view:

```json
[
  {"id": "A1", "polygon": [[100, 80], [300, 80], [300, 260], [100, 260]]},
  {"id": "A2", "x": 320, "y": 80, "w": 200, "h": 180}
]
```

## Two modes

**Real mode** (`--source`) runs YOLOv8 (Ultralytics) on a video file or
webcam and detects the COCO vehicle classes: car, truck, bus, motorcycle.
The same assignment/overlay code then grades the bays. Any YOLOv8 weights
can be supplied with `--model`.

**Simulation mode** (`--simulate`) renders a top-down parking lot with
OpenCV — two rows of marked bays, driving aisles, an entry/exit lane — and
animates cars that drive in, park in a free bay, stay a while, then leave.
The scene publishes its cars as vehicle boxes through the *same detector
interface* the YOLO detector uses, so the occupancy pipeline being demoed
is the real one, not a mock-up. Simulation needs only `opencv-python` and
`numpy` — no model download, no GPU.

## Tech stack

- **Python 3.9+**
- **OpenCV** — rendering, polygon tests (`pointPolygonTest`), overlay, video I/O
- **NumPy** — geometry
- **YOLOv8 / Ultralytics** — real-mode vehicle detection only

## Install

```bash
git clone https://github.com/alishan24456-glitch/smart-parking-occupancy.git
cd smart-parking-occupancy
pip install -r requirements.txt
```

## Usage

Simulation, live preview window:

```bash
python parking_occupancy.py --simulate
```

Simulation, rendered to a video file (headless-friendly):

```bash
python parking_occupancy.py --simulate --output demo.mp4 --no-display --duration 30
```

Real footage or webcam:

```bash
python parking_occupancy.py --source parking_lot.mp4
python parking_occupancy.py --source parking_lot.mp4 --output annotated.mp4
python parking_occupancy.py --source 0                       # webcam
python parking_occupancy.py --source parking_lot.mp4 --spots bays.json
python parking_occupancy.py --source parking_lot.mp4 --rows 3 --cols 8
```

Useful options: `--conf 0.4` (detection threshold), `--model yolov8s.pt`,
`--stable-frames 6` (state-change debounce), `--seed 7` (simulation
randomness). Press `q` or `Esc` to stop a preview. Every state change is
logged, and a run ends with a summary line, e.g.
`[summary] total=12 occupied=7 free=5 occupancy=58.3% peak_occupied=8`.

## Demo

The demo video for this project (`smart-parking-occupancy-demo.mp4`) is
the **simulation mode rendered through the actual occupancy pipeline** —
the cars are drawn by the scene, but every FREE/OCCUPIED decision, bay
outline and HUD number is produced by the same code that grades real
footage. It is a simulation, not real parking-lot footage.

## Real-world uses

- Mall / office parking guidance: drive "spaces available" signs and
  per-floor free-bay counts
- Occupancy analytics: utilisation by hour/day for pricing and staffing
- Enforcement support: spotting bays occupied far beyond a time limit
- EV / disabled bay monitoring when combined with bay-type metadata

## Honest limitations

- The centre-in-bay rule assumes a roughly top-down or mildly angled
  camera; a very oblique view needs accurate per-bay polygons (use
  `--spots`) and can still mis-assign vehicles that straddle two bays.
- Real-mode quality depends on the YOLOv8 model and the camera view —
  no accuracy figures are claimed here; they would have to be measured
  on a specific lot's footage.
- The default grid is a starting layout, not a calibration: real lots
  should define their bays with `--spots`.

## Future improvements

- **Per-bay calibration UI**: click the corners of each bay on a paused
  frame and save the polygons to JSON automatically
- **Automatic perspective calibration** from detected stall lines
- **Plate recognition** on occupied bays for enforcement/permit checks
- Dwell-time tracking per bay and a daily utilisation report (CSV/chart)
- MQTT/REST output so signs and apps can consume the counts live

## Author

**Abdul Hanan** — Computer Vision & AI Developer
GitHub: [alishan24456-glitch](https://github.com/alishan24456-glitch)
LinkedIn: https://www.linkedin.com/in/ali-raza-computervision/
