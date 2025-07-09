"""
Player Detection Module
Uses YOLOv11 model to detect players in video frames
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any
import torch


class PlayerDetector:
    """Player detection using YOLOv11 model"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        """
        Initialize the player detector
        
        Args:
            model_path: Path to the YOLOv11 model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self._load_model()
    
    def _load_model(self):
        """Load the YOLOv11 model"""
        try:
            self.model = YOLO(self.model_path)
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def detect_players(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect players in a single frame
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            List of detection dictionaries with keys: bbox, confidence, class_id
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Run inference
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence and class
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Filter for player class (assuming class 0 is player)
                    if class_id == 0:  # Adjust based on your model's class mapping
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class_id': class_id
                        }
                        detections.append(detection)
        
        return detections
    
    def get_detection_features(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Extract features from a detected player for re-identification
        
        Args:
            frame: Input frame
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Feature vector for the player
        """
        x1, y1, x2, y2 = bbox
        
        # Extract player region
        player_region = frame[y1:y2, x1:x2]
        
        if player_region.size == 0:
            return np.zeros(512)  # Return zero vector if region is empty
        
        # Resize to standard size
        player_region = cv2.resize(player_region, (64, 128))
        
        # Convert to grayscale for simplicity
        gray = cv2.cvtColor(player_region, cv2.COLOR_BGR2GRAY)
        
        # Simple feature extraction using HOG-like approach
        # In a real implementation, you'd use a proper ReID backbone
        features = self._extract_simple_features(gray)
        
        return features
    
    def _extract_simple_features(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Extract simple features from grayscale image
        This is a simplified version - in production, use proper ReID models
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Feature vector
        """
        # Simple histogram-based features
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize
        
        # Add some spatial information
        h, w = gray_image.shape
        top_half = gray_image[:h//2, :]
        bottom_half = gray_image[h//2:, :]
        
        top_hist = cv2.calcHist([top_half], [0], None, [128], [0, 256]).flatten() / (h//2 * w)
        bottom_hist = cv2.calcHist([bottom_half], [0], None, [128], [0, 256]).flatten() / (h//2 * w)
        
        # Combine features
        features = np.concatenate([hist, top_hist, bottom_hist])
        
        return features 