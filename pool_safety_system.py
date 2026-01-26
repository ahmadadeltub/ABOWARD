

import cv2
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum
import time
import sys
from scipy.spatial import distance

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QListWidget, QFileDialog,
    QMessageBox, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Global configuration parameters for the pool safety system."""
    
    # Detection settings
    CONFIDENCE_THRESHOLD = 0.5
    CHILD_HEIGHT_RATIO = 0.6
    
    # Tracking settings
    MAX_TRACK_AGE = 30
    MIN_HITS = 3
    
    # Zone settings
    POOL_APPROACH_DISTANCE = 100
    
    # Drowning detection
    DROWNING_TIME_THRESHOLD = 5.0
    DROWNING_MOVEMENT_THRESHOLD = 15
    
    # Jump detection
    JUMP_VELOCITY_THRESHOLD = 50
    
    # Display colors (BGR format)
    COLOR_POOL_BOUNDARY = (0, 255, 255)  # Yellow
    COLOR_SWIMMING = (255, 200, 0)  # Cyan
    COLOR_STANDING = (0, 255, 0)  # Green
    COLOR_DROWNING = (0, 0, 255)  # Red
    COLOR_CHILD_WARNING = (0, 0, 255)  # Red text for child warnings
    COLOR_ALERT_BG = (0, 0, 100)  # Dark red background
    COLOR_CHILD_ALERT_BG = (100, 50, 0)  # Dark blue background (BGR)
    
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2


# ==============================================================================
# ENUMS AND DATA CLASSES
# ==============================================================================

class PersonState(Enum):
    UNKNOWN = "Unknown"
    STANDING = "Standing on Deck"
    SWIMMING = "Swimming"
    DROWNING = "Drowning Detected"
    APPROACHING = "Approaching Pool"


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    @property
    def bottom_center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, self.y2)
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class TrackedPerson:
    track_id: int
    bbox: BoundingBox
    confidence: float
    is_child: bool = False
    state: PersonState = PersonState.UNKNOWN
    in_pool: bool = False
    position_history: deque = field(default_factory=lambda: deque(maxlen=60))
    last_movement_time: float = field(default_factory=time.time)
    entered_pool: bool = False
    exited_pool: bool = False
    velocity: Tuple[float, float] = (0.0, 0.0)
    just_jumped: bool = False


# ==============================================================================
# POOL ZONE MANAGER
# ==============================================================================

class PoolZoneManager:
    def __init__(self):
        self.pool_polygon: Optional[np.ndarray] = None
        self.is_selecting = False
        self.current_points: List[Tuple[int, int]] = []
        self.selection_complete = False
        
    def start_selection(self):
        self.current_points = []
        self.is_selecting = True
        self.selection_complete = False
        
    def add_point(self, x: int, y: int):
        if self.is_selecting:
            self.current_points.append((x, y))
            
    def complete_selection(self):
        if len(self.current_points) >= 3:
            self.pool_polygon = np.array(self.current_points, dtype=np.int32)
            self.selection_complete = True
        self.is_selecting = False
        
    def reset(self):
        self.pool_polygon = None
        self.is_selecting = False
        self.current_points = []
        self.selection_complete = False
        
    def is_point_in_pool(self, point: Tuple[int, int]) -> bool:
        if self.pool_polygon is None:
            return False
        result = cv2.pointPolygonTest(self.pool_polygon, point, False)
        return result >= 0
    
    def get_distance_to_pool(self, point: Tuple[int, int]) -> float:
        if self.pool_polygon is None:
            return float('inf')
        return abs(cv2.pointPolygonTest(self.pool_polygon, point, True))
    
    def draw_zone(self, frame: np.ndarray):
        if self.pool_polygon is not None:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [self.pool_polygon], (0, 255, 255))
            cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
            cv2.polylines(frame, [self.pool_polygon], True, Config.COLOR_POOL_BOUNDARY, 3)
        elif self.is_selecting and len(self.current_points) > 0:
            pts = np.array(self.current_points, dtype=np.int32)
            cv2.polylines(frame, [pts], False, (0, 255, 0), 2)
            for point in self.current_points:
                cv2.circle(frame, point, 5, (0, 255, 0), -1)


# ==============================================================================
# SIMPLE TRACKER
# ==============================================================================

class SimpleTracker:
    def __init__(self, max_age: int = 30, min_hits: int = 3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: Dict[int, TrackedPerson] = {}
        self.next_id = 1
        self.frame_count = 0
        
    def _iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        x1, y1 = max(box1.x1, box2.x1), max(box1.y1, box2.y1)
        x2, y2 = min(box1.x2, box2.x2), min(box1.y2, box2.y2)
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        union_area = box1.area + box2.area - inter_area
        return inter_area / union_area if union_area > 0 else 0
    
    def reset(self):
        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0
    
    def update(self, detections: List[Tuple[BoundingBox, float, bool]]) -> Dict[int, TrackedPerson]:
        self.frame_count += 1
        
        if not detections:
            to_remove = [tid for tid, t in self.tracks.items() 
                        if self.frame_count - getattr(t, 'last_seen', self.frame_count) > self.max_age]
            for tid in to_remove:
                del self.tracks[tid]
            return self.tracks
        
        matched_tracks = set()
        
        for bbox, conf, is_child in detections:
            best_iou, best_track_id = 0.3, None
            
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                iou = self._iou(bbox, track.bbox)
                if iou > best_iou:
                    best_iou, best_track_id = iou, track_id
            
            if best_track_id is not None:
                old_center = self.tracks[best_track_id].bbox.center
                velocity = (bbox.center[0] - old_center[0], bbox.center[1] - old_center[1])
                self.tracks[best_track_id].bbox = bbox
                self.tracks[best_track_id].confidence = conf
                self.tracks[best_track_id].is_child = is_child
                self.tracks[best_track_id].velocity = velocity
                self.tracks[best_track_id].position_history.append(bbox.center)
                setattr(self.tracks[best_track_id], 'last_seen', self.frame_count)
                matched_tracks.add(best_track_id)
            else:
                new_track = TrackedPerson(track_id=self.next_id, bbox=bbox, confidence=conf, is_child=is_child)
                new_track.position_history.append(bbox.center)
                setattr(new_track, 'last_seen', self.frame_count)
                self.tracks[self.next_id] = new_track
                self.next_id += 1
        
        to_remove = [tid for tid in self.tracks if tid not in matched_tracks 
                    and self.frame_count - getattr(self.tracks[tid], 'last_seen', self.frame_count) > self.max_age]
        for tid in to_remove:
            del self.tracks[tid]
            
        return self.tracks


# ==============================================================================
# ACTIVITY CLASSIFIER
# ==============================================================================

class ActivityClassifier:
    def __init__(self, pool_zone: PoolZoneManager):
        self.pool_zone = pool_zone
        self.movement_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
        self.last_movement_time: Dict[int, float] = {}
        
    def reset(self):
        self.movement_history = defaultdict(lambda: deque(maxlen=30))
        self.last_movement_time = {}
        
    def classify(self, person: TrackedPerson) -> PersonState:
        center = person.bbox.bottom_center
        in_pool = self.pool_zone.is_point_in_pool(center)
        
        if len(person.position_history) >= 2:
            movement = distance.euclidean(person.position_history[-2], person.position_history[-1])
            self.movement_history[person.track_id].append(movement)
            if movement > Config.DROWNING_MOVEMENT_THRESHOLD:
                self.last_movement_time[person.track_id] = time.time()
        
        if person.track_id not in self.last_movement_time:
            self.last_movement_time[person.track_id] = time.time()
        
        if in_pool:
            time_since_movement = time.time() - self.last_movement_time[person.track_id]
            if time_since_movement > Config.DROWNING_TIME_THRESHOLD:
                if len(self.movement_history[person.track_id]) >= 10:
                    avg_movement = np.mean(list(self.movement_history[person.track_id])[-10:])
                    if avg_movement < Config.DROWNING_MOVEMENT_THRESHOLD / 2:
                        return PersonState.DROWNING
            return PersonState.SWIMMING
        else:
            distance_to_pool = self.pool_zone.get_distance_to_pool(center)
            if distance_to_pool < Config.POOL_APPROACH_DISTANCE:
                return PersonState.APPROACHING
            return PersonState.STANDING
    
    def detect_jump(self, person: TrackedPerson, was_in_pool: bool, is_in_pool: bool) -> bool:
        if not was_in_pool and is_in_pool:
            if abs(person.velocity[1]) > Config.JUMP_VELOCITY_THRESHOLD:
                return True
            if len(person.position_history) >= 2:
                movement = distance.euclidean(person.position_history[-2], person.position_history[-1])
                if movement > Config.JUMP_VELOCITY_THRESHOLD:
                    return True
        return False


# ==============================================================================
# ALERT MANAGER
# ==============================================================================

class AlertManager:
    def __init__(self):
        self.active_alerts: List[Tuple[str, float, Tuple[int, int, int]]] = []
        self.alert_duration = 3.0
        self.alert_log: List[Tuple[str, str]] = []
        
    def add_alert(self, message: str, color: Tuple[int, int, int] = (0, 0, 255)):
        self.active_alerts.append((message, time.time(), color))
        self.alert_log.append((time.strftime("%H:%M:%S"), message))
        if len(self.alert_log) > 50:
            self.alert_log.pop(0)
        
    def update(self):
        current_time = time.time()
        self.active_alerts = [(m, t, c) for m, t, c in self.active_alerts if current_time - t < self.alert_duration]
        
    def reset(self):
        self.active_alerts = []
        self.alert_log = []
        
    def draw(self, frame: np.ndarray, start_y: int = 50):
        self.update()
        for i, (message, _, color) in enumerate(self.active_alerts[-5:]):
            y = start_y + i * 40
            text_size = cv2.getTextSize(message, Config.FONT, 0.8, 2)[0]
            
            # Use dark blue background for child-related alerts
            if "Child" in message or "CHILD" in message:
                bg_color = Config.COLOR_CHILD_ALERT_BG
                text_color = (0, 0, 255)  # Red text
            else:
                bg_color = Config.COLOR_ALERT_BG
                text_color = color
            
            cv2.rectangle(frame, (10, y - 25), (20 + text_size[0], y + 10), bg_color, -1)
            cv2.putText(frame, message, (15, y), Config.FONT, 0.8, text_color, 2)


# ==============================================================================
# STATISTICS TRACKER
# ==============================================================================

class StatisticsTracker:
    def __init__(self):
        self.total_entered = 0
        self.total_exited = 0
        self.current_in_pool = 0
        self.tracked_states: Dict[int, bool] = {}
        
    def reset(self):
        self.total_entered = 0
        self.total_exited = 0
        self.current_in_pool = 0
        self.tracked_states = {}
        
    def update(self, tracks: Dict[int, TrackedPerson], pool_zone: PoolZoneManager):
        currently_in_pool = 0
        for track_id, person in tracks.items():
            is_in_pool = pool_zone.is_point_in_pool(person.bbox.bottom_center)
            was_in_pool = self.tracked_states.get(track_id, False)
            if is_in_pool:
                currently_in_pool += 1
            if is_in_pool and not was_in_pool:
                self.total_entered += 1
            if not is_in_pool and was_in_pool:
                self.total_exited += 1
            self.tracked_states[track_id] = is_in_pool
        self.current_in_pool = currently_in_pool
        current_ids = set(tracks.keys())
        self.tracked_states = {k: v for k, v in self.tracked_states.items() if k in current_ids}


# ==============================================================================
# PERSON DETECTOR
# ==============================================================================

class PersonDetector:
    def __init__(self, model_path: str = 'yolov8n.pt'):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model_loaded = True
            print(f"[INFO] YOLOv8 model loaded: {model_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load YOLO model: {e}")
            self.model = None
            self.model_loaded = False
        self.height_samples: List[int] = []
        
    def detect(self, frame: np.ndarray) -> List[Tuple[BoundingBox, float, bool]]:
        detections = []
        if not self.model_loaded:
            return detections
        results = self.model(frame, verbose=False, classes=[0])
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < Config.CONFIDENCE_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = BoundingBox(x1, y1, x2, y2)
                is_child = self._is_child(bbox)
                detections.append((bbox, conf, is_child))
        return detections
    
    def _is_child(self, bbox: BoundingBox) -> bool:
        height = bbox.height
        self.height_samples.append(height)
        if len(self.height_samples) > 100:
            self.height_samples.pop(0)
        if len(self.height_samples) < 10:
            return False
        sorted_heights = sorted(self.height_samples)
        adult_ref = sorted_heights[int(len(sorted_heights) * 0.75)]
        return height < adult_ref * Config.CHILD_HEIGHT_RATIO


# ==============================================================================
# VISUALIZER
# ==============================================================================

class Visualizer:
    @staticmethod
    def draw_person(frame: np.ndarray, person: TrackedPerson):
        bbox = person.bbox
        color_map = {
            PersonState.SWIMMING: Config.COLOR_SWIMMING,
            PersonState.STANDING: Config.COLOR_STANDING,
            PersonState.DROWNING: Config.COLOR_DROWNING,
            PersonState.APPROACHING: Config.COLOR_CHILD_WARNING,
            PersonState.UNKNOWN: (200, 200, 200)
        }
        color = color_map.get(person.state, (200, 200, 200))
        thickness = 3 if person.state == PersonState.DROWNING else 2
        cv2.rectangle(frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, thickness)
        label_parts = [f"ID:{person.track_id}"]
        if person.is_child:
            label_parts.append("CHILD")
        label_parts.append(person.state.value)
        label = " | ".join(label_parts)
        text_size = cv2.getTextSize(label, Config.FONT, 0.5, 1)[0]
        cv2.rectangle(frame, (bbox.x1, bbox.y1 - 20), (bbox.x1 + text_size[0] + 5, bbox.y1), color, -1)
        cv2.putText(frame, label, (bbox.x1 + 2, bbox.y1 - 5), Config.FONT, 0.5, (0, 0, 0), 1)
        cv2.circle(frame, bbox.bottom_center, 4, color, -1)


# ==============================================================================
# CLICKABLE VIDEO LABEL
# ==============================================================================

class ClickableLabel(QLabel):
    clicked = pyqtSignal(int, int)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(event.x(), event.y())
        super().mousePressEvent(event)


# ==============================================================================
# MAIN GUI APPLICATION
# ==============================================================================

class PoolSafetyGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🏊 Pool Safety Monitoring System")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(self.get_stylesheet())
        
        # Video capture
        self.cap = None
        self.video_source = 0
        self.is_running = False
        self.is_paused = False
        
        # Components
        self.detector = None
        self.pool_zone = PoolZoneManager()
        self.tracker = SimpleTracker(max_age=Config.MAX_TRACK_AGE, min_hits=Config.MIN_HITS)
        self.classifier = None
        self.alert_manager = AlertManager()
        self.stats = StatisticsTracker()
        self.visualizer = Visualizer()
        self.previous_pool_states: Dict[int, bool] = {}
        
        # Timer for video updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video)
        
        # Scale factors and offset for click coordinates
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.video_width = 640
        self.video_height = 480
        self.displayed_width = 640
        self.displayed_height = 480
        
        # Current frame for zone selection preview
        self.current_frame = None
        
        self.build_gui()
        
    def get_stylesheet(self):
        return """
            QMainWindow { background-color: #1a1a2e; }
            QGroupBox { 
                font-weight: bold; 
                font-size: 14px;
                color: #4cc9f0; 
                border: 2px solid #4361ee;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                font-size: 13px;
                font-weight: bold;
                padding: 12px 20px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover { opacity: 0.8; }
            QLabel { color: white; font-size: 12px; }
            QListWidget {
                background-color: #0f0f23;
                color: #ffd166;
                font-family: Consolas;
                font-size: 11px;
                border: 1px solid #4361ee;
                border-radius: 5px;
            }
        """
        
    def build_gui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title_label = QLabel("🏊 Pool Safety Monitoring System")
        title_label.setFont(QFont('Helvetica', 28, QFont.Bold))
        title_label.setStyleSheet("color: #4cc9f0; padding: 15px; background-color: #16213e; border-radius: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Content area
        content_layout = QHBoxLayout()
        
        # Left - Video
        left_panel = QFrame()
        left_panel.setStyleSheet("background-color: #0f0f23; border-radius: 10px; padding: 5px;")
        left_layout = QVBoxLayout(left_panel)
        
        self.video_label = ClickableLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #0a0a1a; border: 2px solid #4361ee; border-radius: 8px;")
        self.video_label.clicked.connect(self.on_video_click)
        left_layout.addWidget(self.video_label)
        
        self.status_label = QLabel("Status: Ready - Select video source to begin")
        self.status_label.setStyleSheet("color: #7f8c8d; font-size: 13px; padding: 8px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.status_label)
        
        # Safety status indicators
        safety_frame = QFrame()
        safety_frame.setStyleSheet("background-color: #16213e; border-radius: 8px; padding: 10px;")
        safety_layout = QHBoxLayout(safety_frame)
        safety_layout.setSpacing(20)
        
        # Main safety status indicator
        self.safety_indicator = QLabel("MONITORING OFF")
        self.safety_indicator.setStyleSheet("""
            background-color: #6c757d; 
            color: white; 
            font-size: 18px; 
            font-weight: bold; 
            padding: 15px 30px; 
            border-radius: 8px;
        """)
        self.safety_indicator.setAlignment(Qt.AlignCenter)
        safety_layout.addWidget(self.safety_indicator, stretch=2)
        
        # Pool status
        self.pool_status = QLabel("Pool: Empty")
        self.pool_status.setStyleSheet("""
            background-color: #0f3460; 
            color: #4cc9f0; 
            font-size: 14px; 
            font-weight: bold; 
            padding: 10px 20px; 
            border-radius: 5px;
        """)
        self.pool_status.setAlignment(Qt.AlignCenter)
        safety_layout.addWidget(self.pool_status, stretch=1)
        
        # Drowning alert indicator
        self.drowning_indicator = QLabel("NO DROWNING")
        self.drowning_indicator.setStyleSheet("""
            background-color: #06d6a0; 
            color: white; 
            font-size: 14px; 
            font-weight: bold; 
            padding: 10px 20px; 
            border-radius: 5px;
        """)
        self.drowning_indicator.setAlignment(Qt.AlignCenter)
        safety_layout.addWidget(self.drowning_indicator, stretch=1)
        
        left_layout.addWidget(safety_frame)
        
        content_layout.addWidget(left_panel, stretch=3)
        
        # Right - Controls
        right_panel = QFrame()
        right_panel.setFixedWidth(350)
        right_panel.setStyleSheet("background-color: #16213e; border-radius: 10px;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(15)
        right_layout.setContentsMargins(15, 15, 15, 15)
        
        # Source selection
        source_group = QGroupBox(" Video Source ")
        source_layout = QVBoxLayout(source_group)
        
        btn_webcam = QPushButton("Use Webcam")
        btn_webcam.setStyleSheet("background-color: #4361ee; color: white;")
        btn_webcam.setToolTip("Use your computer's webcam as the video source")
        btn_webcam.clicked.connect(self.use_webcam)
        source_layout.addWidget(btn_webcam)
        
        btn_file = QPushButton("Open Video File")
        btn_file.setStyleSheet("background-color: #7209b7; color: white;")
        btn_file.setToolTip("Open a video file (MP4, AVI, MOV, MKV)")
        btn_file.clicked.connect(self.open_video_file)
        source_layout.addWidget(btn_file)
        
        right_layout.addWidget(source_group)
        
        # Pool zone
        zone_group = QGroupBox(" Pool Zone ")
        zone_layout = QVBoxLayout(zone_group)
        
        self.btn_select_zone = QPushButton("Define Pool Area")
        self.btn_select_zone.setStyleSheet("background-color: #2ec4b6; color: white;")
        self.btn_select_zone.setToolTip("Click to start drawing the pool boundary on the video")
        self.btn_select_zone.clicked.connect(self.start_zone_selection)
        zone_layout.addWidget(self.btn_select_zone)
        
        self.btn_complete_zone = QPushButton("Complete Selection")
        self.btn_complete_zone.setStyleSheet("background-color: #06d6a0; color: white;")
        self.btn_complete_zone.setToolTip("Finish defining the pool area (requires at least 3 points)")
        self.btn_complete_zone.setEnabled(False)
        self.btn_complete_zone.clicked.connect(self.complete_zone_selection)
        zone_layout.addWidget(self.btn_complete_zone)
        
        btn_reset = QPushButton("Reset Zone")
        btn_reset.setStyleSheet("background-color: #ff6b35; color: white;")
        btn_reset.setToolTip("Clear the pool zone and start over")
        btn_reset.clicked.connect(self.reset_zone)
        zone_layout.addWidget(btn_reset)
        
        right_layout.addWidget(zone_group)
        
        # Playback controls
        ctrl_group = QGroupBox(" Playback Controls ")
        ctrl_layout = QHBoxLayout(ctrl_group)
        
        self.btn_start = QPushButton("START")
        self.btn_start.setStyleSheet("background-color: #06d6a0; color: white; font-size: 14px; font-weight: bold; padding: 15px;")
        self.btn_start.setToolTip("Start monitoring the pool for people")
        self.btn_start.clicked.connect(self.start_monitoring)
        ctrl_layout.addWidget(self.btn_start)
        
        self.btn_pause = QPushButton("PAUSE")
        self.btn_pause.setStyleSheet("background-color: #ffd166; color: black; font-size: 14px; font-weight: bold; padding: 15px;")
        self.btn_pause.setToolTip("Pause/Resume the video feed")
        self.btn_pause.clicked.connect(self.toggle_pause)
        ctrl_layout.addWidget(self.btn_pause)
        
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setStyleSheet("background-color: #e63946; color: white; font-size: 14px; font-weight: bold; padding: 15px;")
        self.btn_stop.setToolTip("Stop monitoring completely")
        self.btn_stop.clicked.connect(self.stop_monitoring)
        ctrl_layout.addWidget(self.btn_stop)
        
        right_layout.addWidget(ctrl_group)
        
        # Statistics
        stats_group = QGroupBox(" 📊 Statistics ")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stat_in_pool = QLabel("In Pool: 0")
        self.stat_in_pool.setStyleSheet("background-color: #0f3460; color: #4cc9f0; font-size: 16px; font-weight: bold; padding: 10px; border-radius: 5px;")
        self.stat_in_pool.setAlignment(Qt.AlignCenter)
        self.stat_in_pool.setToolTip("Current number of people inside the pool area")
        stats_layout.addWidget(self.stat_in_pool)
        
        self.stat_entered = QLabel("Total Entered: 0")
        self.stat_entered.setStyleSheet("background-color: #0f3460; color: #06d6a0; font-size: 16px; font-weight: bold; padding: 10px; border-radius: 5px;")
        self.stat_entered.setAlignment(Qt.AlignCenter)
        self.stat_entered.setToolTip("Total number of people who entered the pool")
        stats_layout.addWidget(self.stat_entered)
        
        self.stat_exited = QLabel("Total Exited: 0")
        self.stat_exited.setStyleSheet("background-color: #0f3460; color: #ff6b35; font-size: 16px; font-weight: bold; padding: 10px; border-radius: 5px;")
        self.stat_exited.setAlignment(Qt.AlignCenter)
        self.stat_exited.setToolTip("Total number of people who exited the pool")
        stats_layout.addWidget(self.stat_exited)
        
        btn_clear_counters = QPushButton("Clear Counters")
        btn_clear_counters.setStyleSheet("background-color: #4361ee; color: white;")
        btn_clear_counters.setToolTip("Reset all entry/exit counters to zero")
        btn_clear_counters.clicked.connect(self.clear_counters)
        stats_layout.addWidget(btn_clear_counters)
        
        right_layout.addWidget(stats_group)
        
        # Alerts
        alert_group = QGroupBox(" Recent Alerts ")
        alert_layout = QVBoxLayout(alert_group)
        
        self.alert_list = QListWidget()
        self.alert_list.setMaximumHeight(150)
        self.alert_list.setToolTip("Log of safety alerts (drowning, child approaching, etc.)")
        alert_layout.addWidget(self.alert_list)
        
        btn_clear_alerts = QPushButton("Clear Alerts")
        btn_clear_alerts.setStyleSheet("background-color: #4361ee; color: white;")
        btn_clear_alerts.setToolTip("Clear all alerts from the log")
        btn_clear_alerts.clicked.connect(self.clear_alerts)
        alert_layout.addWidget(btn_clear_alerts)
        
        right_layout.addWidget(alert_group)
        
        # Exit button
        btn_exit = QPushButton("Exit Application")
        btn_exit.setStyleSheet("background-color: #e63946; color: white;")
        btn_exit.setToolTip("Close the application")
        btn_exit.clicked.connect(self.close)
        right_layout.addWidget(btn_exit)
        
        right_layout.addStretch()
        content_layout.addWidget(right_panel)
        
        main_layout.addLayout(content_layout)
        
    def use_webcam(self):
        self.video_source = 0
        self.status_label.setText("Status: Webcam selected - Press Start to begin")
        self.initialize_video_capture()
        
    def open_video_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", 
                                                   "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*.*)")
        if filepath:
            self.video_source = filepath
            self.status_label.setText("Status: Video loaded - Press Start to begin")
            self.initialize_video_capture()
            
    def initialize_video_capture(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Cannot open video source!")
            return
        if self.detector is None:
            self.detector = PersonDetector('yolov8n.pt')
        self.classifier = ActivityClassifier(self.pool_zone)
        
        # Get video dimensions
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
            
    def start_zone_selection(self):
        if self.cap is None:
            QMessageBox.warning(self, "Warning", "Please select a video source first!")
            return
        self.pool_zone.start_selection()
        self.btn_select_zone.setStyleSheet("background-color: #ffd166; color: black;")
        self.btn_select_zone.setText("🖱️ Click to add points...")
        self.btn_complete_zone.setEnabled(True)
        self.status_label.setText("Status: Click on the video to define pool boundary points")
        
    def complete_zone_selection(self):
        self.pool_zone.complete_selection()
        if self.pool_zone.selection_complete:
            self.btn_select_zone.setStyleSheet("background-color: #2ec4b6; color: white;")
            self.btn_select_zone.setText("✏️ Define Pool Area")
            self.btn_complete_zone.setEnabled(False)
            self.status_label.setText("Status: Pool zone defined - Ready to monitor")
            QMessageBox.information(self, "Success", "Pool zone defined successfully!")
        else:
            QMessageBox.warning(self, "Warning", "Need at least 3 points to define pool area!")
            
    def reset_zone(self):
        self.pool_zone.reset()
        self.stats.reset()
        self.tracker.reset()
        if self.classifier:
            self.classifier.reset()
        self.alert_manager.reset()
        self.previous_pool_states = {}
        self.btn_select_zone.setStyleSheet("background-color: #2ec4b6; color: white;")
        self.btn_select_zone.setText("✏️ Define Pool Area")
        self.btn_complete_zone.setEnabled(False)
        self.status_label.setText("Status: Pool zone reset - Define new zone")
        self.update_stats_display()
        self.alert_list.clear()
    
    def clear_counters(self):
        """Clear the entry/exit counters without resetting the pool zone."""
        self.stats.reset()
        self.update_stats_display()
        self.status_label.setText("Status: Counters cleared")
        
    def clear_alerts(self):
        """Clear the alert log."""
        self.alert_manager.reset()
        self.alert_list.clear()
        self.status_label.setText("Status: Alerts cleared")
        
    def on_video_click(self, x: int, y: int):
        if self.pool_zone.is_selecting and self.current_frame is not None:
            # Adjust for the offset (centering) within the label
            adjusted_x = x - self.offset_x
            adjusted_y = y - self.offset_y
            
            # Check if click is within the video area
            if 0 <= adjusted_x < self.displayed_width and 0 <= adjusted_y < self.displayed_height:
                # Scale coordinates from display to actual video
                scaled_x = int(adjusted_x * self.scale_x)
                scaled_y = int(adjusted_y * self.scale_y)
                self.pool_zone.add_point(scaled_x, scaled_y)
                print(f"[DEBUG] Added point: ({scaled_x}, {scaled_y}) - Total points: {len(self.pool_zone.current_points)}")
                
                # Update the display to show the point
                self.display_frame(self.current_frame)
            
    def start_monitoring(self):
        if self.cap is None:
            QMessageBox.warning(self, "Warning", "Please select a video source first!")
            return
        if not self.pool_zone.selection_complete:
            QMessageBox.warning(self, "Warning", "Please define the pool area first!")
            return
        self.is_running = True
        self.is_paused = False
        self.status_label.setText("Status: 🟢 Monitoring active...")
        self.timer.start(30)
        
    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.status_label.setText("Status: ⏸️ Paused")
        else:
            self.status_label.setText("Status: 🟢 Monitoring active...")
            
    def stop_monitoring(self):
        self.is_running = False
        self.is_paused = False
        self.timer.stop()
        self.status_label.setText("Status: ⏹️ Stopped")
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        output_frame = frame.copy()
        
        if self.pool_zone.selection_complete and self.detector:
            detections = self.detector.detect(frame)
            tracks = self.tracker.update(detections)
            
            for track_id, person in tracks.items():
                was_in_pool = self.previous_pool_states.get(track_id, False)
                is_in_pool = self.pool_zone.is_point_in_pool(person.bbox.bottom_center)
                person.state = self.classifier.classify(person)
                person.in_pool = is_in_pool
                
                if person.is_child:
                    if self.classifier.detect_jump(person, was_in_pool, is_in_pool):
                        self.alert_manager.add_alert("CHILD JUMPED INTO POOL!", Config.COLOR_DROWNING)
                    if person.state == PersonState.APPROACHING:
                        self.alert_manager.add_alert("Child Approaching Pool", Config.COLOR_CHILD_WARNING)
                
                if person.state == PersonState.DROWNING:
                    self.alert_manager.add_alert(f"🚨 DROWNING DETECTED - ID:{track_id}", Config.COLOR_DROWNING)
                
                self.previous_pool_states[track_id] = is_in_pool
                self.visualizer.draw_person(output_frame, person)
            
            self.stats.update(tracks, self.pool_zone)
            current_ids = set(tracks.keys())
            self.previous_pool_states = {k: v for k, v in self.previous_pool_states.items() if k in current_ids}
        
        self.pool_zone.draw_zone(output_frame)
        self.alert_manager.draw(output_frame)
        
        return output_frame
        
    def update_video(self):
        if not self.is_running:
            return
            
        if not self.is_paused:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    self.stop_monitoring()
                    return
            
            processed_frame = self.process_frame(frame)
            self.display_frame(processed_frame)
            self.update_stats_display()
            self.update_alert_log()
            self.update_safety_indicators()
            
    def display_frame(self, frame: np.ndarray):
        # Store the current frame for zone selection preview
        self.current_frame = frame.copy()
        
        # Draw zone points if selecting
        display_frame = frame.copy()
        self.pool_zone.draw_zone(display_frame)
        
        label_w = self.video_label.width()
        label_h = self.video_label.height()
        
        if label_w > 1 and label_h > 1:
            h, w = display_frame.shape[:2]
            scale = min(label_w / w, label_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Store scale factors for click coordinate conversion
            self.scale_x = w / new_w
            self.scale_y = h / new_h
            self.displayed_width = new_w
            self.displayed_height = new_h
            
            # Calculate offset for centering
            self.offset_x = (label_w - new_w) // 2
            self.offset_y = (label_h - new_h) // 2
            
            display_frame = cv2.resize(display_frame, (new_w, new_h))
        
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))
        
    def update_stats_display(self):
        self.stat_in_pool.setText(f"In Pool: {self.stats.current_in_pool}")
        self.stat_entered.setText(f"Total Entered: {self.stats.total_entered}")
        self.stat_exited.setText(f"Total Exited: {self.stats.total_exited}")
        
    def update_alert_log(self):
        current_alerts = len(self.alert_manager.alert_log)
        displayed_alerts = self.alert_list.count()
        if current_alerts > displayed_alerts:
            for i in range(displayed_alerts, current_alerts):
                time_str, msg = self.alert_manager.alert_log[i]
                self.alert_list.addItem(f"[{time_str}] {msg}")
                self.alert_list.scrollToBottom()
    
    def update_safety_indicators(self):
        """Update the safety status indicators based on current pool state."""
        # Check if anyone is drowning
        drowning_detected = False
        for track_id, person in self.tracker.tracks.items():
            if person.state == PersonState.DROWNING:
                drowning_detected = True
                break
        
        # Update drowning indicator
        if drowning_detected:
            self.drowning_indicator.setText("DROWNING!")
            self.drowning_indicator.setStyleSheet("""
                background-color: #e63946; 
                color: white; 
                font-size: 14px; 
                font-weight: bold; 
                padding: 10px 20px; 
                border-radius: 5px;
            """)
        else:
            self.drowning_indicator.setText("NO DROWNING")
            self.drowning_indicator.setStyleSheet("""
                background-color: #06d6a0; 
                color: white; 
                font-size: 14px; 
                font-weight: bold; 
                padding: 10px 20px; 
                border-radius: 5px;
            """)
        
        # Update pool status
        people_in_pool = self.stats.current_in_pool
        if people_in_pool == 0:
            self.pool_status.setText("Pool: Empty")
            self.pool_status.setStyleSheet("""
                background-color: #0f3460; 
                color: #4cc9f0; 
                font-size: 14px; 
                font-weight: bold; 
                padding: 10px 20px; 
                border-radius: 5px;
            """)
        else:
            self.pool_status.setText(f"Pool: {people_in_pool} People")
            self.pool_status.setStyleSheet("""
                background-color: #4361ee; 
                color: white; 
                font-size: 14px; 
                font-weight: bold; 
                padding: 10px 20px; 
                border-radius: 5px;
            """)
        
        # Update main safety indicator
        if drowning_detected:
            self.safety_indicator.setText("DANGER - DROWNING DETECTED!")
            self.safety_indicator.setStyleSheet("""
                background-color: #e63946; 
                color: white; 
                font-size: 18px; 
                font-weight: bold; 
                padding: 15px 30px; 
                border-radius: 8px;
            """)
        elif people_in_pool > 0:
            self.safety_indicator.setText("ALL SAFE - MONITORING ACTIVE")
            self.safety_indicator.setStyleSheet("""
                background-color: #06d6a0; 
                color: white; 
                font-size: 18px; 
                font-weight: bold; 
                padding: 15px 30px; 
                border-radius: 8px;
            """)
        else:
            self.safety_indicator.setText("MONITORING ACTIVE")
            self.safety_indicator.setStyleSheet("""
                background-color: #4361ee; 
                color: white; 
                font-size: 18px; 
                font-weight: bold; 
                padding: 15px 30px; 
                border-radius: 8px;
            """)
                
    def closeEvent(self, event):
        self.is_running = False
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        event.accept()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "="*60)
    print("POOL SAFETY MONITORING SYSTEM - GUI MODE")
    print("="*60)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(26, 26, 46))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(15, 15, 35))
    palette.setColor(QPalette.AlternateBase, QColor(26, 26, 46))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(67, 97, 238))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(67, 97, 238))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    window = PoolSafetyGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
