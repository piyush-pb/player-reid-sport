"""
Player Tracking Module
Handles player tracking across frames and re-identification
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment
import cv2


class PlayerTrack:
    """Represents a tracked player"""
    
    def __init__(self, track_id: int, bbox: List[int], features: np.ndarray, frame_id: int):
        """
        Initialize a player track
        
        Args:
            track_id: Unique track ID
            bbox: Bounding box [x1, y1, x2, y2]
            features: Feature vector
            frame_id: Frame where player was first detected
        """
        self.track_id = track_id
        self.bbox = bbox
        self.features = features
        self.frame_id = frame_id
        self.last_seen = frame_id
        self.history = [bbox]
        self.feature_history = [features]
        self.missed_frames = 0
        self.max_missed_frames = 30  # Maximum frames to keep track when player is not detected
    
    def update(self, bbox: List[int], features: np.ndarray, frame_id: int):
        """Update track with new detection"""
        self.bbox = bbox
        self.features = features
        self.last_seen = frame_id
        self.history.append(bbox)
        self.feature_history.append(features)
        self.missed_frames = 0
    
    def predict(self):
        """Predict next position based on history (simple linear prediction)"""
        if len(self.history) >= 2:
            # Simple linear prediction
            prev_bbox = self.history[-1]
            prev_prev_bbox = self.history[-2]
            
            # Calculate velocity
            vx = (prev_bbox[0] - prev_prev_bbox[0]) / 2
            vy = (prev_bbox[1] - prev_prev_bbox[1]) / 2
            
            # Predict next position
            predicted_bbox = [
                int(prev_bbox[0] + vx),
                int(prev_bbox[1] + vy),
                int(prev_bbox[2] + vx),
                int(prev_bbox[3] + vy)
            ]
            
            return predicted_bbox
        return self.bbox
    
    def is_active(self, current_frame: int) -> bool:
        """Check if track is still active"""
        return (current_frame - self.last_seen) <= self.max_missed_frames
    
    def get_average_features(self) -> np.ndarray:
        """Get average features from history"""
        if self.feature_history:
            return np.mean(self.feature_history, axis=0)
        return self.features


class PlayerTracker:
    """Player tracking and re-identification system"""
    
    def __init__(self, 
                 iou_threshold: float = 0.3,
                 feature_threshold: float = 0.7,
                 max_missed_frames: int = 30):
        """
        Initialize the player tracker
        
        Args:
            iou_threshold: IoU threshold for matching detections to tracks
            feature_threshold: Feature similarity threshold for re-identification
            max_missed_frames: Maximum frames to keep inactive tracks
        """
        self.iou_threshold = iou_threshold
        self.feature_threshold = feature_threshold
        self.max_missed_frames = max_missed_frames
        
        self.tracks: List[PlayerTrack] = []
        self.next_track_id = 1
        self.frame_count = 0
    
    def update(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Update tracks with new detections
        
        Args:
            detections: List of detections from the detector
            frame: Current frame for feature extraction
            
        Returns:
            List of tracked players with IDs
        """
        self.frame_count += 1
        
        # Remove inactive tracks
        self.tracks = [track for track in self.tracks if track.is_active(self.frame_count)]
        
        # Extract features for new detections
        detection_features = []
        for detection in detections:
            features = self._extract_features(frame, detection['bbox'])
            detection_features.append(features)
        
        # Match detections to existing tracks
        matched_tracks, matched_detections = self._match_detections_to_tracks(
            detections, detection_features
        )
        
        # Update matched tracks
        for track_idx, det_idx in zip(matched_tracks, matched_detections):
            self.tracks[track_idx].update(
                detections[det_idx]['bbox'],
                detection_features[det_idx],
                self.frame_count
            )
        
        # Create new tracks for unmatched detections
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_detections]
        for det_idx in unmatched_detections:
            new_track = PlayerTrack(
                self.next_track_id,
                detections[det_idx]['bbox'],
                detection_features[det_idx],
                self.frame_count
            )
            self.tracks.append(new_track)
            self.next_track_id += 1
        
        # Return tracked players
        tracked_players = []
        for track in self.tracks:
            if track.is_active(self.frame_count):
                tracked_players.append({
                    'track_id': track.track_id,
                    'bbox': track.bbox,
                    'confidence': 1.0,  # We don't have confidence from tracking
                    'class_id': 0  # Assuming all are players
                })
        
        return tracked_players
    
    def _extract_features(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Extract features from a detection (placeholder - should use detector's method)"""
        x1, y1, x2, y2 = bbox
        
        # Extract player region
        player_region = frame[y1:y2, x1:x2]
        
        if player_region.size == 0:
            return np.zeros(512)
        
        # Resize to standard size
        player_region = cv2.resize(player_region, (64, 128))
        
        # Convert to grayscale
        gray = cv2.cvtColor(player_region, cv2.COLOR_BGR2GRAY)
        
        # Simple feature extraction
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # Add spatial information
        h, w = gray.shape
        top_half = gray[:h//2, :]
        bottom_half = gray[h//2:, :]
        
        top_hist = cv2.calcHist([top_half], [0], None, [128], [0, 256]).flatten() / (h//2 * w)
        bottom_hist = cv2.calcHist([bottom_half], [0], None, [128], [0, 256]).flatten() / (h//2 * w)
        
        features = np.concatenate([hist, top_hist, bottom_hist])
        
        return features
    
    def _match_detections_to_tracks(self, 
                                   detections: List[Dict[str, Any]], 
                                   detection_features: List[np.ndarray]) -> Tuple[List[int], List[int]]:
        """
        Match detections to existing tracks using IoU and feature similarity
        
        Args:
            detections: List of detections
            detection_features: List of feature vectors for detections
            
        Returns:
            Tuple of (matched_track_indices, matched_detection_indices)
        """
        if not self.tracks or not detections:
            return [], []
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, detection in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou(track.bbox, detection['bbox'])
        
        # Calculate feature similarity matrix
        feature_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            track_features = track.get_average_features()
            for j, det_features in enumerate(detection_features):
                # Cosine similarity (1 - cosine distance)
                similarity = 1 - cosine(track_features, det_features)
                feature_matrix[i, j] = similarity
        
        # Combine IoU and feature similarity
        # Weight IoU more heavily for spatial matching
        combined_matrix = 0.7 * iou_matrix + 0.3 * feature_matrix
        
        # Apply thresholds
        iou_mask = iou_matrix >= self.iou_threshold
        feature_mask = feature_matrix >= self.feature_threshold
        combined_mask = iou_mask | feature_mask
        
        # Apply mask to combined matrix
        masked_matrix = combined_matrix * combined_mask
        
        # Use Hungarian algorithm for optimal assignment
        track_indices, det_indices = linear_sum_assignment(-masked_matrix)
        
        # Filter out low-confidence matches
        matched_tracks = []
        matched_detections = []
        
        for track_idx, det_idx in zip(track_indices, det_indices):
            if masked_matrix[track_idx, det_idx] > 0:
                matched_tracks.append(track_idx)
                matched_detections.append(det_idx)
        
        return matched_tracks, matched_detections
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_track_history(self, track_id: int) -> List[List[int]]:
        """Get bounding box history for a specific track"""
        for track in self.tracks:
            if track.track_id == track_id:
                return track.history
        return [] 