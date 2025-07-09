import json
import os
import sys
import matplotlib.pyplot as plt

# Add src directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import visualizer

# Load results
with open('results/tracking_results.json', 'r') as f:
    results = json.load(f)

frame_results = results['frame_results']

# Generate and save player count plot
fig = visualizer.plot_player_count_over_time(frame_results)
fig.savefig('results/player_count_over_time.png', dpi=300, bbox_inches='tight')
plt.close(fig)

print('Player count over time plot saved to results/player_count_over_time.png') 