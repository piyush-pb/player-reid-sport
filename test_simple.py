#!/usr/bin/env python3
"""
Simple test script for player re-identification system
Tests basic functionality without requiring the full pipeline
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

def test_opencv():
    """Test OpenCV installation"""
    print("Testing OpenCV...")
    try:
        # Create a simple test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[25:75, 25:75] = [0, 255, 0]  # Green rectangle
        
        # Test basic operations
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        print("✓ OpenCV is working correctly")
        return True
    except Exception as e:
        print(f"✗ OpenCV test failed: {e}")
        return False

def test_numpy():
    """Test NumPy installation"""
    print("Testing NumPy...")
    try:
        # Test basic operations
        arr = np.array([1, 2, 3, 4, 5])
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        
        print("✓ NumPy is working correctly")
        return True
    except Exception as e:
        print(f"✗ NumPy test failed: {e}")
        return False

def test_video_io():
    """Test video I/O functionality"""
    print("Testing video I/O...")
    try:
        # Create a simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('test_output.mp4', fourcc, 30.0, (640, 480))
        
        # Write a few frames
        for i in range(30):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :, 0] = i * 8  # Blue channel
            out.write(frame)
        
        out.release()
        
        # Clean up
        if os.path.exists('test_output.mp4'):
            os.remove('test_output.mp4')
        
        print("✓ Video I/O is working correctly")
        return True
    except Exception as e:
        print(f"✗ Video I/O test failed: {e}")
        return False

def test_directory_structure():
    """Test if required directories exist"""
    print("Testing directory structure...")
    
    required_dirs = ['data', 'models', 'src', 'output', 'docs']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"✗ Missing directories: {missing_dirs}")
        return False
    else:
        print("✓ All required directories exist")
        return True

def test_requirements():
    """Test if required packages are installed"""
    print("Testing required packages...")
    
    required_packages = [
        'cv2',
        'numpy',
        'ultralytics',
        'torch',
        'matplotlib',
        'tqdm',
        'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'ultralytics':
                import ultralytics
            elif package == 'torch':
                import torch
            elif package == 'matplotlib':
                import matplotlib
            elif package == 'tqdm':
                import tqdm
            elif package == 'scipy':
                import scipy
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"✗ Missing packages: {missing_packages}")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False
    else:
        print("✓ All required packages are installed")
        return True

def main():
    """Run all tests"""
    print("=== Player Re-Identification System - Simple Tests ===\n")
    
    tests = [
        test_opencv,
        test_numpy,
        test_video_io,
        test_directory_structure,
        test_requirements
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! The system is ready to use.")
        return True
    else:
        print("✗ Some tests failed. Please fix the issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 