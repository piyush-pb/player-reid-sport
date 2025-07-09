#!/usr/bin/env python3
"""
Setup script for Player Re-Identification System
Liat.ai AI Intern Assignment
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data', 'models', 'output', 'docs', 'results']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Directory '{directory}' ready")

def download_instructions():
    """Print download instructions"""
    print("\n📥 Download Instructions:")
    print("1. Download the YOLOv11 model from:")
    print("   https://drive.google.com/file/d/1-5fOSHOSB9UxyP_enOOZNAMScrePvCMD/view")
    print("   Save it as 'models/yolov11.pt'")
    print()
    print("2. Download the test video from the assignment materials:")
    print("   Save '15sec_input_720p.mp4' as 'data/15sec_input_720p.mp4'")
    print()

def test_installation():
    """Test the installation"""
    print("Testing installation...")
    try:
        result = subprocess.run([sys.executable, "test_simple.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Installation test passed")
            return True
        else:
            print("❌ Installation test failed")
            print("Output:", result.stdout)
            print("Errors:", result.stderr)
            return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("=== Player Re-Identification System Setup ===\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\nCreating directories...")
    create_directories()
    
    # Install requirements
    print("\nInstalling requirements...")
    if not install_requirements():
        print("Please install requirements manually: pip install -r requirements.txt")
    
    # Download instructions
    download_instructions()
    
    # Test installation
    print("Running installation test...")
    if test_installation():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Download the model and video files (see instructions above)")
        print("2. Run: python main.py --input data/15sec_input_720p.mp4 --model models/yolov11.pt --output output/result.mp4")
        print("3. Check the output video and results in the output/ and results/ directories")
    else:
        print("\n⚠️  Setup completed with warnings. Please check the errors above.")
        print("You may need to install some packages manually.")

if __name__ == "__main__":
    main() 