# /app/ml_models/object_detection.py
"""
ObjectDetector — YOLOv8 based cell phone / prohibited item detection.

PyTorch 2.6 Fix:
  torch.load() now defaults to weights_only=True, which breaks YOLO's
  custom ultralytics class deserialization.
  Fix: register ultralytics globals as safe BEFORE importing YOLO model.
"""

import os
import sys
import logging

logger = logging.getLogger('EagleEye')

# ── PyTorch 2.6 safe-globals patch ───────────────────────────────────────────
# Must be done BEFORE ultralytics loads the model weights.
def _patch_torch_safe_globals():
    """Allow ultralytics model classes through PyTorch 2.6+ weights_only check."""
    try:
        import torch
        torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
        if torch_version >= (2, 6):
            try:
                # Preferred: add_safe_globals API (PyTorch >= 2.4)
                from ultralytics.nn.tasks import DetectionModel, SegmentationModel, ClassificationModel
                import torch.serialization as tser
                tser.add_safe_globals([DetectionModel, SegmentationModel, ClassificationModel])
                logger.info("[YOLO] Registered ultralytics globals for PyTorch 2.6+")
            except (ImportError, AttributeError):
                # Fallback: monkey-patch torch.load to use weights_only=False
                import torch
                _original_load = torch.load
                def _patched_load(*args, **kwargs):
                    kwargs.setdefault('weights_only', False)
                    return _original_load(*args, **kwargs)
                torch.load = _patched_load
                logger.info("[YOLO] Applied torch.load weights_only=False patch for PyTorch 2.6+")
    except Exception as e:
        logger.warning(f"[YOLO] Could not apply PyTorch 2.6 patch: {e}")

_patch_torch_safe_globals()

# ── Now safe to import ultralytics ───────────────────────────────────────────
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except Exception as e:
    logger.error(f"[YOLO] ultralytics import failed: {e}")
    _YOLO_AVAILABLE = False

# ── Model path resolution ─────────────────────────────────────────────────────
# Try local paths in order of preference
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CANDIDATE_PATHS = [
    os.path.join(_BASE_DIR, 'yolov8n.pt'),
    os.path.join(_BASE_DIR, 'app', 'yolov8n.pt'),
    os.path.join(_BASE_DIR, 'app', 'ml_models', 'yolov8n.pt'),
    'yolov8n.pt',   # Let ultralytics download if not found locally
]

YOLO_MODEL_PATH = next((p for p in _CANDIDATE_PATHS if os.path.exists(p)), 'yolov8n.pt')

# Which COCO class IDs to flag as prohibited items
TARGET_CLASSES = {
    67: 'cell phone',
    73: 'book',        # open book during closed-book exam
    63: 'laptop',
    76: 'scissors',
}


class ObjectDetector:
    """
    YOLOv8-based prohibited object detector.
    Gracefully degrades if the model cannot be loaded.
    """

    def __init__(self, model_path: str = None):
        self.model = None
        self._available = False

        if not _YOLO_AVAILABLE:
            logger.error("[YOLO] ultralytics not available — object detection disabled.")
            return

        mp = model_path or YOLO_MODEL_PATH
        try:
            self.model = YOLO(mp)
            self._available = True
            logger.info(f"[YOLO] ObjectDetector ready — model: {mp}")
        except Exception as e:
            logger.error(f"[YOLO] Failed to load model '{mp}': {e}")
            # Try the monkey-patch fallback one more time
            try:
                import torch
                _orig = torch.load
                def _safe_load(*a, **kw):
                    kw['weights_only'] = False
                    return _orig(*a, **kw)
                torch.load = _safe_load
                self.model = YOLO(mp)
                self._available = True
                logger.info(f"[YOLO] ObjectDetector ready after fallback patch — model: {mp}")
            except Exception as e2:
                logger.error(f"[YOLO] Fallback load also failed: {e2}. Object detection disabled.")

    @property
    def is_available(self) -> bool:
        return self._available

    def detect_objects(self, frame, confidence_threshold: float = 0.45):
        """
        Run YOLO inference on a frame and return prohibited objects.

        Returns:
            list of dicts: [{'label': str, 'confidence': float, 'box': (x1,y1,x2,y2)}]
            Returns [] if model is not available or no targets found.
        """
        if not self._available or self.model is None:
            return []

        try:
            results = self.model.predict(frame, verbose=False, conf=confidence_threshold)
            detected = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    if class_id in TARGET_CLASSES:
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detected.append({
                            'label': TARGET_CLASSES[class_id],
                            'confidence': conf,
                            'box': (x1, y1, x2, y2)
                        })
            return detected
        except Exception as e:
            logger.warning(f"[YOLO] detect_objects error: {e}")
            return []