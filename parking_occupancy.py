#!/usr/bin/env python3
"""
Smart Parking Occupancy Monitor
================================

Reports which parking bays are FREE and which are OCCUPIED, live, from a
single camera-style view of a parking lot.

One shared occupancy pipeline serves both run modes:

* **Real mode** (``--source``) -- YOLOv8 detects vehicles (car / truck /
  bus / motorcycle) in a video file or webcam feed.
* **Simulation mode** (``--simulate``) -- a top-down parking lot is rendered
  with OpenCV; cars drive in, park, stay a while and leave.  The scene hands
  vehicle boxes to the pipeline through exactly the same detector interface
  as the YOLO detector, so the bay-assignment, overlay and counting code is
  identical in both modes.  Simulation needs only OpenCV + NumPy.

How occupancy is decided
------------------------
Each parking bay is a polygon.  A bay counts as OCCUPIED when the **centre
point of a detected vehicle box falls inside the bay polygon** (and FREE
when no vehicle centre is inside).  A short stability window (a few
consecutive frames) prevents flicker when a box jitters on a bay edge.

Examples
--------
    python parking_occupancy.py --simulate
    python parking_occupancy.py --simulate --output demo.mp4 --no-display
    python parking_occupancy.py --source parking_lot.mp4
    python parking_occupancy.py --source 0
    python parking_occupancy.py --source parking_lot.mp4 --spots bays.json

Author: Abdul Hanan -- Computer Vision & AI Developer
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIM_WIDTH, SIM_HEIGHT = 1280, 720   # simulation canvas
SIM_FPS = 30                        # simulation frame rate

VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle"}  # COCO class names

COLOUR_FREE = (0, 190, 70)          # green  (BGR)
COLOUR_OCCUPIED = (0, 55, 230)      # red    (BGR)
COLOUR_BOX = (225, 225, 225)        # vehicle detection boxes
COLOUR_TEXT = (255, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Detections and the detector interface
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """One detected vehicle: an axis-aligned box plus label/confidence."""

    x1: float
    y1: float
    x2: float
    y2: float
    label: str = "car"
    confidence: float = 1.0

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class VehicleDetector:
    """Interface every detector implements: frame in, vehicle boxes out."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        raise NotImplementedError


class YOLOVehicleDetector(VehicleDetector):
    """Real-mode detector backed by YOLOv8 (ultralytics, lazily imported)."""

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.35):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "Real mode needs the 'ultralytics' package for YOLOv8.\n"
                "Install it with:  pip install ultralytics\n"
                "(Simulation mode --simulate does not need it.)"
            ) from exc
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(frame, conf=self.confidence, verbose=False)
        detections: list[Detection] = []
        for result in results:
            names = result.names or {}
            boxes = result.boxes
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), cls_id, conf in zip(xyxy, classes, confs):
                label = str(names.get(cls_id, cls_id))
                if label in VEHICLE_LABELS:
                    detections.append(
                        Detection(float(x1), float(y1), float(x2), float(y2),
                                  label, float(conf))
                    )
        return detections


class SimulatedDetector(VehicleDetector):
    """Simulation-mode detector: forwards the scene's vehicle boxes.

    The scene knows where its cars are; it publishes them through this
    interface in exactly the same shape the YOLO detector produces, which
    is what lets both modes share one occupancy pipeline.
    """

    def __init__(self, scene: "ParkingScene"):
        self.scene = scene

    def detect(self, frame: np.ndarray) -> list[Detection]:  # frame unused
        return self.scene.detections()


# ---------------------------------------------------------------------------
# Parking bays and the lot layout
# ---------------------------------------------------------------------------

@dataclass
class ParkingBay:
    """A single parking bay: a polygon plus its (debounced) state."""

    bay_id: str
    polygon: np.ndarray  # shape (4+, 2), float32
    occupied: bool = False
    _streak: int = field(default=0, repr=False)

    @property
    def rect(self) -> tuple[int, int, int, int]:
        x, y, w, h = cv2.boundingRect(self.polygon.astype(np.int32))
        return int(x), int(y), int(w), int(h)

    @property
    def centre(self) -> tuple[float, float]:
        pts = self.polygon.astype(np.float64)
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())

    def contains_centre(self, point: tuple[float, float]) -> bool:
        return cv2.pointPolygonTest(self.polygon, point, False) >= 0

    def update(self, candidate: bool, stable_frames: int) -> bool:
        """Debounce the raw per-frame candidate; True if the state flipped."""
        if candidate == self.occupied:
            self._streak = 0
            return False
        self._streak += 1
        if self._streak >= max(1, stable_frames):
            self.occupied = candidate
            self._streak = 0
            return True
        return False


@dataclass
class LotLayout:
    """Geometry of the default bay grid (shared by real mode and the sim)."""

    bays: list[ParkingBay]
    lot_x0: int
    lot_x1: int
    lot_y0: int
    lot_y1: int
    road_x: int          # centre x of the vertical access road
    lane_y: int          # centre y of the entry/exit lane
    aisle_ys: list[int]  # driving aisle centre y for each row


def default_layout(width: int, height: int, rows: int, cols: int) -> LotLayout:
    """Build an even grid of bays over the lot region of a frame.

    Layout (top-down): bays occupy the upper lot region in ``rows`` bands
    separated by driving aisles; a vertical access road runs down the left
    side and an entry/exit lane runs along the bottom.
    """
    if rows < 1 or cols < 1:
        raise ValueError("--rows and --cols must both be at least 1")

    lot_x0 = int(0.117 * width)
    lot_x1 = int(0.969 * width)
    lot_y0 = int(0.172 * height)
    lot_y1 = int(0.792 * height)
    road_x = int(0.060 * width)
    lane_y = int(0.903 * height)

    lot_h = lot_y1 - lot_y0
    # Each inter-row aisle is 0.42x a bay's height.
    row_h = lot_h / (rows + 0.42 * (rows - 1))
    gap = 0.42 * row_h
    bay_w = (lot_x1 - lot_x0) / cols

    bays: list[ParkingBay] = []
    aisle_ys: list[int] = []
    for r in range(rows):
        top = lot_y0 + r * (row_h + gap)
        bottom = top + row_h
        if r < rows - 1:
            aisle_ys.append(int(bottom + gap / 2))
        else:
            aisle_ys.append(lane_y)  # the bottom row is served by the lane
        for c in range(cols):
            left = lot_x0 + c * bay_w
            right = left + bay_w
            polygon = np.array(
                [[left, top], [right, top], [right, bottom], [left, bottom]],
                dtype=np.float32,
            )
            bays.append(ParkingBay(f"{chr(65 + r)}{c + 1}", polygon))

    return LotLayout(bays, lot_x0, lot_x1, lot_y0, lot_y1,
                     road_x, lane_y, aisle_ys)


def load_bays_json(path: str) -> list[ParkingBay]:
    """Load custom bays from a JSON file.

    Each entry is either a polygon::

        {"id": "A1", "polygon": [[x, y], [x, y], [x, y], [x, y]]}

    or an axis-aligned rectangle::

        {"id": "A2", "x": 320, "y": 80, "w": 200, "h": 180}

    Coordinates are pixels in the video frame.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, list) or not raw:
        raise ValueError("spots JSON must be a non-empty list of bays")

    bays: list[ParkingBay] = []
    for i, item in enumerate(raw):
        bay_id = str(item.get("id", f"B{i + 1}"))
        if "polygon" in item:
            polygon = np.array(item["polygon"], dtype=np.float32)
            if polygon.ndim != 2 or polygon.shape[0] < 3:
                raise ValueError(f"bay {bay_id}: polygon needs >= 3 points")
        elif all(k in item for k in ("x", "y", "w", "h")):
            x, y, w, h = (float(item[k]) for k in ("x", "y", "w", "h"))
            polygon = np.array(
                [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                dtype=np.float32,
            )
        else:
            raise ValueError(
                f"bay {bay_id}: needs either 'polygon' or x/y/w/h fields")
        bays.append(ParkingBay(bay_id, polygon))
    return bays


# ---------------------------------------------------------------------------
# The shared occupancy pipeline
# ---------------------------------------------------------------------------

class OccupancyMonitor:
    """Assigns vehicle detections to bays and draws the annotated view."""

    def __init__(self, bays: list[ParkingBay], stable_frames: int = 4):
        self.bays = bays
        self.stable_frames = stable_frames
        self.peak_occupied = 0
        self.last_stats: dict = {}

    def process(self, frame: np.ndarray, detections: list[Detection],
                fps: float) -> tuple[np.ndarray, dict, list[str]]:
        """Run one frame through the pipeline.

        Returns (annotated_frame, stats, flip_events).
        """
        events: list[str] = []
        for bay in self.bays:
            candidate = any(bay.contains_centre(d.centre) for d in detections)
            if bay.update(candidate, self.stable_frames):
                state = "OCCUPIED" if bay.occupied else "FREE"
                events.append(f"{bay.bay_id} -> {state}")

        occupied = sum(1 for b in self.bays if b.occupied)
        total = len(self.bays)
        self.peak_occupied = max(self.peak_occupied, occupied)
        stats = {
            "total": total,
            "occupied": occupied,
            "free": total - occupied,
            "percent": (100.0 * occupied / total) if total else 0.0,
            "fps": fps,
        }
        self.last_stats = stats
        return self._draw(frame, detections, stats), stats, events

    # -- drawing ------------------------------------------------------------

    def _draw(self, frame: np.ndarray, detections: list[Detection],
              stats: dict) -> np.ndarray:
        out = frame.copy()

        # Translucent bay fills on a separate overlay, then outlines/labels.
        overlay = out.copy()
        for bay in self.bays:
            colour = COLOUR_OCCUPIED if bay.occupied else COLOUR_FREE
            cv2.fillPoly(overlay, [bay.polygon.astype(np.int32)], colour)
        cv2.addWeighted(overlay, 0.20, out, 0.80, 0, out)

        for bay in self.bays:
            colour = COLOUR_OCCUPIED if bay.occupied else COLOUR_FREE
            cv2.polylines(out, [bay.polygon.astype(np.int32)], True, colour, 2)
            x, y, _w, _h = bay.rect
            state = "OCCUPIED" if bay.occupied else "FREE"
            cv2.putText(out, bay.bay_id, (x + 8, y + 24), FONT, 0.65,
                        COLOUR_TEXT, 2, cv2.LINE_AA)
            cv2.putText(out, state, (x + 8, y + 48), FONT, 0.5,
                        colour, 2, cv2.LINE_AA)

        # Vehicle boxes (the raw stream the occupancy decision came from).
        for det in detections:
            p1 = (int(det.x1), int(det.y1))
            p2 = (int(det.x2), int(det.y2))
            cv2.rectangle(out, p1, p2, COLOUR_BOX, 1)
            cv2.putText(out, det.label, (p1[0], max(12, p1[1] - 4)), FONT,
                        0.45, COLOUR_BOX, 1, cv2.LINE_AA)

        self._draw_hud(out, stats)
        return out

    @staticmethod
    def _draw_hud(img: np.ndarray, stats: dict) -> None:
        panel_w, panel_h = 430, 104
        overlay = img.copy()
        cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.68, img, 0.32, 0, img)
        cv2.rectangle(img, (8, 8), (8 + panel_w, 8 + panel_h), (90, 90, 90), 1)

        cv2.putText(img, "SMART PARKING OCCUPANCY", (20, 36), FONT, 0.66,
                    COLOUR_TEXT, 2, cv2.LINE_AA)
        cv2.putText(
            img,
            f"Total: {stats['total']}   Occupied: {stats['occupied']}   "
            f"Free: {stats['free']}",
            (20, 64), FONT, 0.58, COLOUR_TEXT, 2, cv2.LINE_AA)
        cv2.putText(
            img,
            f"Occupancy: {stats['percent']:.1f}%      FPS: {stats['fps']:.1f}",
            (20, 92), FONT, 0.58, COLOUR_TEXT, 2, cv2.LINE_AA)

    def summary_line(self) -> str:
        s = self.last_stats
        if not s:
            return "[summary] no frames processed"
        return (f"[summary] total={s['total']} occupied={s['occupied']} "
                f"free={s['free']} occupancy={s['percent']:.1f}% "
                f"peak_occupied={self.peak_occupied}")


# ---------------------------------------------------------------------------
# Simulation: a top-down parking lot rendered with OpenCV
# ---------------------------------------------------------------------------

CAR_COLOURS = [
    (60, 60, 200),    # red
    (200, 120, 40),   # blue
    (210, 210, 210),  # silver
    (40, 160, 200),   # orange
    (90, 170, 90),    # green
    (180, 80, 160),   # purple
    (230, 200, 60),   # teal
    (120, 120, 120),  # grey
    (70, 190, 230),   # yellow
    (150, 100, 60),   # steel blue
]

CAR_LENGTH, CAR_WIDTH = 126.0, 74.0
CAR_SPEED = 260.0          # pixels per simulated second
TURN_RATE = 4.2            # radians per simulated second


def _rotate(local_x: float, local_y: float, heading: float) -> tuple[float, float]:
    c, s = math.cos(heading), math.sin(heading)
    return local_x * c - local_y * s, local_x * s + local_y * c


def draw_car(img: np.ndarray, cx: float, cy: float, heading: float,
             colour: tuple[int, int, int]) -> None:
    """Draw a simple but neat top-down car centred at (cx, cy).

    Local frame: +x is the car's front (direction of travel), length runs
    along x, width along y.  The shape is a tapered body with a windshield,
    a roof, rear window, four wheels and two headlights.
    """
    hl, hw = CAR_LENGTH / 2.0, CAR_WIDTH / 2.0

    def pt(lx: float, ly: float) -> tuple[int, int]:
        gx, gy = _rotate(lx, ly, heading)
        return int(cx + gx), int(cy + gy)

    # Wheels (drawn first, slightly proud of the body).
    for lx, ly in ((-hl + 16, -hw - 2), (-hl + 16, hw - 8),
                   (hl - 26, -hw - 2), (hl - 26, hw - 8)):
        wheel = np.array([pt(lx, ly), pt(lx + 12, ly), pt(lx + 12, ly + 10),
                          pt(lx, ly + 10)], dtype=np.int32)
        cv2.fillPoly(img, [wheel], (15, 15, 15))

    # Body: rectangle with a slightly tapered nose.
    body = np.array([
        pt(-hl, -hw + 7), pt(hl - 14, -hw + 4), pt(hl, -hw + 14),
        pt(hl, hw - 14), pt(hl - 14, hw - 4), pt(-hl, hw - 7),
        pt(-hl - 4, 0),
    ], dtype=np.int32)
    cv2.fillPoly(img, [body], colour)
    cv2.polylines(img, [body], True, (20, 20, 20), 2)

    # Windshield (front) and rear window.
    shield = np.array([pt(hl - 34, -hw + 10), pt(hl - 18, -hw + 8),
                       pt(hl - 18, hw - 8), pt(hl - 34, hw - 10)],
                      dtype=np.int32)
    cv2.fillPoly(img, [shield], (60, 45, 30))
    rear = np.array([pt(-hl + 12, -hw + 10), pt(-hl + 24, -hw + 9),
                     pt(-hl + 24, hw - 9), pt(-hl + 12, hw - 10)],
                    dtype=np.int32)
    cv2.fillPoly(img, [rear], (60, 45, 30))

    # Roof (a lighter shade of the body colour).
    roof_colour = tuple(min(255, int(c * 1.25)) for c in colour)
    roof = np.array([pt(-hl + 26, -hw + 9), pt(hl - 38, -hw + 9),
                     pt(hl - 38, hw - 9), pt(-hl + 26, hw - 9)],
                    dtype=np.int32)
    cv2.fillPoly(img, [roof], roof_colour)

    # Headlights.
    for ly in (-hw + 10, hw - 10):
        cv2.circle(img, pt(hl - 2, ly), 4, (200, 240, 255), -1)


@dataclass
class SimCar:
    x: float
    y: float
    heading: float
    colour: tuple[int, int, int]
    waypoints: list[tuple[float, float]]
    state: str = "enter"            # enter -> parked -> leave -> done
    bay_index: int = -1
    park_until: float = 0.0

    def box(self) -> Detection:
        corners = [_rotate(lx, ly, self.heading)
                   for lx, ly in ((-CAR_LENGTH / 2, -CAR_WIDTH / 2),
                                  (-CAR_LENGTH / 2, CAR_WIDTH / 2),
                                  (CAR_LENGTH / 2, -CAR_WIDTH / 2),
                                  (CAR_LENGTH / 2, CAR_WIDTH / 2))]
        xs = [self.x + gx for gx, _ in corners]
        ys = [self.y + gy for _, gy in corners]
        return Detection(min(xs), min(ys), max(xs), max(ys), "car", 1.0)


class ParkingScene:
    """The simulated lot: static geometry plus cars that come and go."""

    def __init__(self, layout: LotLayout, duration: float, seed: int = 42):
        self.layout = layout
        self.duration = duration
        self.rng = random.Random(seed)
        self.time = 0.0
        self.cars: list[SimCar] = []
        self.reserved: set[int] = set()
        self.next_spawn = 0.4
        self.parked_now = 0

    # -- routes --------------------------------------------------------------

    def _enter_route(self, bay_index: int) -> list[tuple[float, float]]:
        lay = self.layout
        bay = lay.bays[bay_index]
        row = bay_index // max(1, (len(lay.bays) // len(lay.aisle_ys)))
        aisle_y = lay.aisle_ys[min(row, len(lay.aisle_ys) - 1)]
        cx, cy = bay.centre
        points = [(-100.0, float(lay.lane_y)), (float(lay.road_x), float(lay.lane_y))]
        if aisle_y != lay.lane_y:
            points.append((float(lay.road_x), float(aisle_y)))
        points.append((cx, float(aisle_y)))
        points.append((cx, cy))
        return points

    def _exit_route(self, bay_index: int) -> list[tuple[float, float]]:
        lay = self.layout
        bay = lay.bays[bay_index]
        row = bay_index // max(1, (len(lay.bays) // len(lay.aisle_ys)))
        aisle_y = lay.aisle_ys[min(row, len(lay.aisle_ys) - 1)]
        cx, cy = bay.centre
        points = [(cx, cy), (cx, float(aisle_y))]
        if aisle_y != lay.lane_y:
            points.append((float(lay.road_x), float(aisle_y)))
            points.append((float(lay.road_x), float(lay.lane_y)))
        points.append((float(SIM_WIDTH + 100), float(lay.lane_y)))
        return points

    # -- behaviour -------------------------------------------------------------

    def _spawn(self) -> None:
        free = [i for i in range(len(self.layout.bays))
                if i not in self.reserved]
        if not free:
            return
        bay_index = self.rng.choice(free)
        self.reserved.add(bay_index)
        route = self._enter_route(bay_index)
        colour = self.rng.choice(CAR_COLOURS)
        self.cars.append(SimCar(route[0][0], route[0][1], 0.0, colour,
                                route[1:], bay_index=bay_index))

    def step(self, dt: float) -> None:
        self.time += dt

        if (self.time >= self.next_spawn
                and self.time < self.duration - 5.0):
            self._spawn()
            self.next_spawn = self.time + self.rng.uniform(1.6, 2.6)

        survivors: list[SimCar] = []
        for car in self.cars:
            if car.state == "parked":
                if self.time >= car.park_until:
                    car.state = "leave"
                    car.waypoints = self._exit_route(car.bay_index)[1:]
                else:
                    survivors.append(car)
                    continue
            if car.waypoints:
                self._drive(car, dt)
                if not car.waypoints:
                    if car.state == "enter":
                        car.state = "parked"
                        car.park_until = self.time + self.rng.uniform(8.0, 16.0)
                    elif car.state == "leave":
                        car.state = "done"
                        self.reserved.discard(car.bay_index)
            if car.state != "done":
                survivors.append(car)
        self.cars = survivors
        self.parked_now = sum(1 for c in self.cars if c.state == "parked")

    @staticmethod
    def _drive(car: SimCar, dt: float) -> None:
        tx, ty = car.waypoints[0]
        dx, dy = tx - car.x, ty - car.y
        dist = math.hypot(dx, dy)
        desired = math.atan2(dy, dx)
        # Turn the nose toward the target (shortest way round).
        diff = (desired - car.heading + math.pi) % (2 * math.pi) - math.pi
        max_turn = TURN_RATE * dt
        car.heading += max(-max_turn, min(max_turn, diff))
        # Slow down a little while turning sharply.
        speed = CAR_SPEED * (0.55 if abs(diff) > 1.2 else 1.0)
        step = speed * dt
        if dist <= max(7.0, step):
            car.x, car.y = tx, ty
            car.waypoints.pop(0)
        else:
            car.x += dx / dist * step
            car.y += dy / dist * step

    # -- frame -----------------------------------------------------------------

    def detections(self) -> list[Detection]:
        return [car.box() for car in self.cars]

    def render(self) -> np.ndarray:
        lay = self.layout
        img = np.full((SIM_HEIGHT, SIM_WIDTH, 3), (46, 47, 50), np.uint8)

        road = (56, 56, 60)
        surface = (64, 63, 60)
        # Lot surface and driving areas.
        cv2.rectangle(img, (lay.lot_x0 - 16, lay.lot_y0 - 14),
                      (lay.lot_x1 + 14, int(0.815 * SIM_HEIGHT)), surface, -1)
        cv2.rectangle(img, (14, lay.lot_y0 - 14),
                      (lay.lot_x0 - 16, SIM_HEIGHT - 12), road, -1)
        cv2.rectangle(img, (0, int(0.845 * SIM_HEIGHT)),
                      (SIM_WIDTH, SIM_HEIGHT - 12), road, -1)
        for y in lay.aisle_ys[:-1]:
            cv2.rectangle(img, (lay.lot_x0 - 16, y - 39),
                          (lay.lot_x1 + 14, y + 39), road, -1)

        # Dashed centre lines on the road, the lane and the aisles.
        dash = (0, 190, 220)
        for x in range(30, SIM_WIDTH - 20, 46):
            cv2.line(img, (x, lay.lane_y), (x + 24, lay.lane_y), dash, 2)
        for y in range(lay.lot_y0, lay.lane_y - 30, 46):
            cv2.line(img, (lay.road_x, y), (lay.road_x, y + 24), dash, 2)
        for y in lay.aisle_ys[:-1]:
            for x in range(lay.lot_x0 + 10, lay.lot_x1 - 20, 46):
                cv2.line(img, (x, y), (x + 24, y), dash, 2)

        # Direction arrows and labels.
        cv2.arrowedLine(img, (lay.road_x, lay.lane_y - 90),
                        (lay.road_x, lay.lane_y - 170), (230, 230, 230), 3,
                        tipLength=0.25)
        cv2.arrowedLine(img, (300, lay.lane_y), (420, lay.lane_y),
                        (230, 230, 230), 3, tipLength=0.25)
        cv2.putText(img, "ENTRY", (24, SIM_HEIGHT - 22), FONT, 0.7,
                    (230, 230, 230), 2, cv2.LINE_AA)
        cv2.putText(img, "EXIT", (SIM_WIDTH - 110, SIM_HEIGHT - 22), FONT,
                    0.7, (230, 230, 230), 2, cv2.LINE_AA)
        cv2.putText(img, "PARKING LOT - TOP-DOWN SIMULATION",
                    (460, 40), FONT, 0.8, (200, 200, 200), 2,
                    cv2.LINE_AA)

        for car in self.cars:
            draw_car(img, car.x, car.y, car.heading, car.colour)
        return img


# ---------------------------------------------------------------------------
# Run loops
# ---------------------------------------------------------------------------

def safe_destroy_windows() -> None:
    """cv2.destroyAllWindows() crashes some headless OpenCV builds; guard it."""
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def make_writer(path: str, fps: float, width: int, height: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"could not open output video for writing: {path}")
    return writer


def run_simulation(args: argparse.Namespace, bays: list[ParkingBay],
                   layout: LotLayout) -> int:
    scene = ParkingScene(layout, duration=args.duration, seed=args.seed)
    detector = SimulatedDetector(scene)
    monitor = OccupancyMonitor(bays, stable_frames=args.stable_frames)

    writer = None
    if args.output:
        writer = make_writer(args.output, SIM_FPS, SIM_WIDTH, SIM_HEIGHT)

    total_frames = int(args.duration * SIM_FPS)
    try:
        for frame_no in range(total_frames):
            scene.step(1.0 / SIM_FPS)
            frame = scene.render()
            detections = detector.detect(frame)
            annotated, _stats, events = monitor.process(frame, detections,
                                                        SIM_FPS)
            for event in events:
                print(f"[event] t={scene.time:5.1f}s  {event}")
            if writer is not None:
                writer.write(annotated)
            if not args.no_display:
                cv2.imshow("Smart Parking Occupancy - simulation", annotated)
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27):
                    print("[info] stopped by user")
                    break
            if frame_no % (SIM_FPS * 5) == 0:
                print(f"[info] t={scene.time:5.1f}s  "
                      f"parked_in_scene={scene.parked_now}")
    finally:
        if writer is not None:
            writer.release()
        if not args.no_display:
            safe_destroy_windows()

    print(monitor.summary_line())
    print(f"[info] scene ground truth at end: parked={scene.parked_now} "
          f"cars_on_screen={len(scene.cars)}")
    return 0


def run_real(args: argparse.Namespace, bays: list[ParkingBay] | None) -> int:
    source: int | str = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"error: could not open source: {args.source}", file=sys.stderr)
        return 2

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0:
        src_fps = 30.0

    if bays is None:
        bays = default_layout(width, height, args.rows, args.cols).bays

    try:
        detector: VehicleDetector = YOLOVehicleDetector(args.model, args.conf)
    except RuntimeError as exc:
        cap.release()
        print(f"error: {exc}", file=sys.stderr)
        return 2

    monitor = OccupancyMonitor(bays, stable_frames=args.stable_frames)
    writer = None
    if args.output:
        writer = make_writer(args.output, src_fps, width, height)

    ema_fps = float(src_fps)
    last = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.perf_counter()
            dt = now - last
            last = now
            if dt > 0:
                ema_fps = 0.9 * ema_fps + 0.1 * (1.0 / dt)

            detections = detector.detect(frame)
            annotated, _stats, events = monitor.process(frame, detections,
                                                        ema_fps)
            for event in events:
                print(f"[event] {event}")
            if writer is not None:
                writer.write(annotated)
            if not args.no_display:
                cv2.imshow("Smart Parking Occupancy", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("[info] stopped by user")
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            safe_destroy_windows()

    print(monitor.summary_line())
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smart parking occupancy monitor: reports which bays "
                    "are FREE vs OCCUPIED, live.",
        epilog="examples:\n"
               "  python parking_occupancy.py --simulate\n"
               "  python parking_occupancy.py --simulate --output demo.mp4 "
               "--no-display\n"
               "  python parking_occupancy.py --source parking_lot.mp4\n"
               "  python parking_occupancy.py --source 0\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_argument_group("mode (choose one)")
    mode.add_argument("--simulate", action="store_true",
                      help="run the built-in top-down parking-lot simulation "
                           "(OpenCV + NumPy only)")
    mode.add_argument("--source", metavar="SRC",
                      help="real mode: video file path or webcam index")

    parser.add_argument("--spots", metavar="JSON",
                        help="JSON file with custom bay polygons/rects "
                             "(overrides the default grid)")
    parser.add_argument("--rows", type=int, default=2,
                        help="default-grid bay rows (default: 2)")
    parser.add_argument("--cols", type=int, default=6,
                        help="default-grid bay columns (default: 6)")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLOv8 weights for real mode (default: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="YOLO confidence threshold (default: 0.35)")
    parser.add_argument("--stable-frames", type=int, default=4,
                        help="frames a bay must agree before its state flips "
                             "(default: 4)")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="simulation length in seconds (default: 30)")
    parser.add_argument("--seed", type=int, default=42,
                        help="simulation random seed (default: 42)")
    parser.add_argument("--output", metavar="FILE",
                        help="save the annotated video to FILE")
    parser.add_argument("--no-display", action="store_true",
                        help="do not open a preview window (headless)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.simulate and args.source is not None:
        parser.error("choose only one mode: --simulate OR --source")
    if not args.simulate and args.source is None:
        parser.error("choose a mode: --simulate or --source <video file|webcam index>")
    if args.rows < 1 or args.cols < 1:
        parser.error("--rows and --cols must be at least 1")
    if args.simulate and args.duration <= 0:
        parser.error("--duration must be positive")

    if args.source is not None and not str(args.source).isdigit():
        if not os.path.exists(args.source):
            parser.error(f"source file not found: {args.source}")

    # Custom bays from JSON, if requested.
    custom_bays: list[ParkingBay] | None = None
    if args.spots:
        if not os.path.exists(args.spots):
            parser.error(f"spots file not found: {args.spots}")
        try:
            custom_bays = load_bays_json(args.spots)
        except (ValueError, json.JSONDecodeError, OSError) as exc:
            parser.error(f"could not load spots file: {exc}")

    if args.simulate:
        layout = default_layout(SIM_WIDTH, SIM_HEIGHT, args.rows, args.cols)
        bays = custom_bays if custom_bays is not None else layout.bays
        return run_simulation(args, bays, layout)
    return run_real(args, custom_bays)


if __name__ == "__main__":
    sys.exit(main())
