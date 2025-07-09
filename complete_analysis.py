#!/usr/bin/env python3
"""
Complete Analysis Script
Generates all analysis plots and summary from existing tracking results
"""

import json
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.visualizer import ResultsVisualizer

def main():
    """Generate complete analysis from existing results"""
    print("=== Generating Complete Analysis ===\n")
    
    # Load results
    results_file = "results/tracking_results.json"
    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        print("Please run the main pipeline first to generate results.")
        return
    
    print("Loading tracking results...")
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    frame_results = results['frame_results']
    track_data = results['track_data']
    
    print(f"Loaded {len(frame_results)} frames with {len(track_data)} tracks")
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Initialize visualizer
    results_visualizer = ResultsVisualizer()
    
    # Generate player count plot
    print("Generating player count over time plot...")
    try:
        fig1 = results_visualizer.plot_player_count_over_time(frame_results)
        fig1.savefig("results/player_count_over_time.png", dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print("✅ Player count plot saved: results/player_count_over_time.png")
    except Exception as e:
        print(f"❌ Error generating player count plot: {e}")
    
    # Generate track positions plot
    print("Generating track positions plot...")
    try:
        # Estimate video dimensions from bounding boxes
        all_bboxes = []
        for frame_result in frame_results:
            for player in frame_result['tracked_players']:
                all_bboxes.append(player['bbox'])
        
        if all_bboxes:
            max_x = max(bbox[2] for bbox in all_bboxes)
            max_y = max(bbox[3] for bbox in all_bboxes)
            video_width = max_x
            video_height = max_y
        else:
            video_width, video_height = 1280, 720  # Default based on your video
        
        fig2 = results_visualizer.plot_track_positions(track_data, video_width, video_height)
        fig2.savefig("results/track_positions.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print("✅ Track positions plot saved: results/track_positions.png")
    except Exception as e:
        print(f"❌ Error generating track positions plot: {e}")
    
    # Generate tracking summary
    print("Generating tracking summary...")
    try:
        summary_text = results_visualizer.create_tracking_summary(track_data, len(frame_results))
        
        with open("results/tracking_summary.txt", 'w') as f:
            f.write(summary_text)
        
        print("✅ Tracking summary saved: results/tracking_summary.txt")
        
        # Also print summary to console
        print("\n" + "="*50)
        print("TRACKING SUMMARY")
        print("="*50)
        print(summary_text)
        
    except Exception as e:
        print(f"❌ Error generating tracking summary: {e}")
    
    # Generate additional statistics
    print("Generating additional statistics...")
    try:
        stats = {
            'total_tracks': len(track_data),
            'total_frames': len(frame_results),
            'track_lengths': {},
            'track_durations': {},
            'average_positions': {}
        }
        
        for track_id, bboxes in track_data.items():
            if len(bboxes) == 0:
                continue
            
            # Track length
            stats['track_lengths'][track_id] = len(bboxes)
            
            # Track duration (assuming 25 FPS from your video)
            stats['track_durations'][track_id] = len(bboxes) / 25.0
            
            # Average position
            centers_x = [(bbox[0] + bbox[2]) / 2 for bbox in bboxes]
            centers_y = [(bbox[1] + bbox[3]) / 2 for bbox in bboxes]
            avg_x = sum(centers_x) / len(centers_x)
            avg_y = sum(centers_y) / len(centers_y)
            stats['average_positions'][track_id] = (avg_x, avg_y)
        
        # Save statistics
        with open("results/tracking_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("✅ Tracking statistics saved: results/tracking_statistics.json")
        
        # Print key statistics
        if stats['track_lengths']:
            avg_track_length = sum(stats['track_lengths'].values()) / len(stats['track_lengths'])
            longest_track = max(stats['track_lengths'].values())
            print(f"📊 Average track length: {avg_track_length:.1f} frames")
            print(f"📊 Longest track: {longest_track} frames")
            print(f"📊 Total unique players: {stats['total_tracks']}")
        
    except Exception as e:
        print(f"❌ Error generating statistics: {e}")
    
    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("- results/player_count_over_time.png")
    print("- results/track_positions.png") 
    print("- results/tracking_summary.txt")
    print("- results/tracking_statistics.json")
    print("\nYou can now view these files to analyze the tracking results!")

if __name__ == "__main__":
    main() 