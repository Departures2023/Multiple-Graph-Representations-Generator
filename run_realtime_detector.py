"""
Quick launcher for real-time graph detection with webcam
This provides TRUE real-time detection with live video feed
"""

import sys
import os

# Add the image_to_description directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'image_to_description'))

from realtime_graph_detector import RealtimeGraphDetector

if __name__ == "__main__":
    print("\n" + "="*70)
    print("REAL-TIME GRAPH DETECTOR - Live Webcam Feed")
    print("="*70)
    print("\nThis will open your webcam and detect graphs in real-time!")
    print("\nControls:")
    print("  SPACE   - Capture and detect current frame")
    print("  'a'     - Toggle auto-detect (continuous detection)")
    print("  'd'     - Toggle arrow detection (directed/undirected)")
    print("  '+'     - Increase node size detection")
    print("  '-'     - Decrease node size detection")
    print("  ']'     - Increase edge sensitivity (detect more edges)")
    print("  '['     - Decrease edge sensitivity (detect fewer edges)")
    print("  'r'     - Show current graph representation in console")
    print("  'q'     - Quit")
    print("\n" + "="*70)
    print("\nStarting webcam... Press 'a' to enable continuous detection!")
    print("="*70 + "\n")
    
    detector = RealtimeGraphDetector()
    detector.run_webcam()


