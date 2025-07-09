"""
Visualization Module
Handles drawing bounding boxes and player IDs on video frames
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from matplotlib.figure import Figure


class VideoVisualizer:
    """Visualizes player tracking results on video frames"""
    
    def __init__(self, colors: Optional[List[Tuple[int, int, int]]] = None):
        """
        Initialize the visualizer
        
        Args:
            colors: List of BGR colors for different player IDs
        """
        if colors is None:
            # Default colors for player IDs (BGR format)
            self.colors: List[Tuple[int, int, int]] = [
                (255, 0, 0),    # Blue
                (0, 255, 0),    # Green
                (0, 0, 255),    # Red
                (255, 255, 0),  # Cyan
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Yellow
                (128, 0, 0),    # Dark Blue
                (0, 128, 0),    # Dark Green
                (0, 0, 128),    # Dark Red
                (128, 128, 0),  # Olive
            ]
        else:
            self.colors = colors
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw detection bounding boxes on frame
        
        Args:
            frame: Input frame
            detections: List of detections with bbox, confidence, class_id
            
        Returns:
            Frame with drawn bounding boxes
        """
        frame_copy = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection.get('confidence', 0.0)
            class_id = detection.get('class_id', 0)
            
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence text
            conf_text = f"{confidence:.2f}"
            cv2.putText(frame_copy, conf_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame_copy
    
    def draw_tracks(self, frame: np.ndarray, tracked_players: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw tracked players with IDs on frame
        
        Args:
            frame: Input frame
            tracked_players: List of tracked players with track_id, bbox
            
        Returns:
            Frame with drawn tracks and IDs
        """
        frame_copy = frame.copy()
        
        for player in tracked_players:
            track_id = player['track_id']
            bbox = player['bbox']
            
            x1, y1, x2, y2 = bbox
            
            # Get color for this track ID
            color = self.colors[(track_id - 1) % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID
            id_text = f"ID: {track_id}"
            cv2.putText(frame_copy, id_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame_copy, (center_x, center_y), 3, color, -1)
        
        return frame_copy
    
    def draw_track_history(self, frame: np.ndarray, track_history: List[List[int]], 
                          track_id: int) -> np.ndarray:
        """
        Draw track history as a trail
        
        Args:
            frame: Input frame
            track_history: List of bounding boxes representing track history
            track_id: Track ID for color selection
            
        Returns:
            Frame with track history drawn
        """
        frame_copy = frame.copy()
        
        if len(track_history) < 2:
            return frame_copy
        
        # Get color for this track
        color = self.colors[(track_id - 1) % len(self.colors)]
        
        # Draw trail
        for i in range(1, len(track_history)):
            prev_bbox = track_history[i - 1]
            curr_bbox = track_history[i]
            
            prev_center = ((prev_bbox[0] + prev_bbox[2]) // 2, 
                          (prev_bbox[1] + prev_bbox[3]) // 2)
            curr_center = ((curr_bbox[0] + curr_bbox[2]) // 2, 
                          (curr_bbox[1] + curr_bbox[3]) // 2)
            
            # Draw line with decreasing opacity
            alpha = 0.3 * (i / len(track_history))
            cv2.line(frame_copy, prev_center, curr_center, color, 2)
        
        return frame_copy
    
    def add_frame_info(self, frame: np.ndarray, frame_number: int, 
                      num_players: int, fps: float = 30.0) -> np.ndarray:
        """
        Add frame information overlay
        
        Args:
            frame: Input frame
            frame_number: Current frame number
            num_players: Number of players detected
            fps: Video FPS
            
        Returns:
            Frame with information overlay
        """
        frame_copy = frame.copy()
        
        # Add frame number
        frame_text = f"Frame: {frame_number}"
        cv2.putText(frame_copy, frame_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add time
        time_seconds = frame_number / fps
        time_text = f"Time: {time_seconds:.2f}s"
        cv2.putText(frame_copy, time_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add player count
        player_text = f"Players: {num_players}"
        cv2.putText(frame_copy, player_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame_copy


class ResultsVisualizer:
    """Visualizes tracking results and statistics"""
    
    def __init__(self):
        """Initialize the results visualizer"""
        pass
    
    def plot_track_positions(self, track_data: Dict[int, List[List[int]]], 
                           video_width: int, video_height: int) -> Figure:
        """
        Plot player track positions over time
        
        Args:
            track_data: Dictionary mapping track_id to list of bounding boxes
            video_width: Video width in pixels
            video_height: Video height in pixels
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot each track
        for track_id, bboxes in track_data.items():
            if len(bboxes) == 0:
                continue
            
            # Extract center points
            centers_x = []
            centers_y = []
            
            for bbox in bboxes:
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                centers_x.append(center_x)
                centers_y.append(center_y)
            
            # Plot track
            ax.plot(centers_x, centers_y, 'o-', label=f'Player {track_id}', 
                   markersize=4, linewidth=2)
            
            # Mark start and end points
            if centers_x:
                ax.plot(centers_x[0], centers_y[0], 'go', markersize=8, label=f'Start {track_id}' if track_id == 1 else "")
                ax.plot(centers_x[-1], centers_y[-1], 'ro', markersize=8, label=f'End {track_id}' if track_id == 1 else "")
        
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_title('Player Track Positions Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set axis limits to video dimensions
        ax.set_xlim(0, video_width)
        ax.set_ylim(video_height, 0)  # Invert Y axis for image coordinates
        
        return fig
    
    def plot_player_count_over_time(self, frame_data: List[Dict[str, Any]]) -> Figure:
        """
        Plot number of players detected over time
        
        Args:
            frame_data: List of frame data with player counts
            
        Returns:
            Matplotlib figure
        """
        frames = list(range(len(frame_data)))
        player_counts = [len(frame['players']) for frame in frame_data]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(frames, player_counts, 'b-', linewidth=2)
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Number of Players')
        ax.set_title('Player Count Over Time')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def create_tracking_summary(self, track_data: Dict[int, List[List[int]]], 
                              total_frames: int) -> str:
        """
        Create a text summary of tracking results
        
        Args:
            track_data: Dictionary mapping track_id to list of bounding boxes
            total_frames: Total number of frames in video
            
        Returns:
            Summary text
        """
        summary = "=== Player Tracking Summary ===\n\n"
        
        summary += f"Total frames processed: {total_frames}\n"
        summary += f"Total unique players tracked: {len(track_data)}\n\n"
        
        for track_id, bboxes in track_data.items():
            if len(bboxes) == 0:
                continue
            
            # Calculate track statistics
            track_length = len(bboxes)
            track_duration = track_length / 30.0  # Assuming 30 FPS
            
            # Calculate average position
            centers_x = [(bbox[0] + bbox[2]) / 2 for bbox in bboxes]
            centers_y = [(bbox[1] + bbox[3]) / 2 for bbox in bboxes]
            avg_x = sum(centers_x) / len(centers_x)
            avg_y = sum(centers_y) / len(centers_y)
            
            summary += f"Player {track_id}:\n"
            summary += f"  - Track length: {track_length} frames ({track_duration:.2f}s)\n"
            summary += f"  - Average position: ({avg_x:.1f}, {avg_y:.1f})\n"
            summary += f"  - First seen: frame {bboxes[0] if bboxes else 0}\n"
            summary += f"  - Last seen: frame {bboxes[-1] if bboxes else 0}\n\n"
        
        return summary 