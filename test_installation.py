"""
Installation Test Script
Tests if all required dependencies are properly installed
"""

import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow: {tf.__version__}")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    try:
        import keras
        print(f"✓ Keras: {keras.__version__}")
    except ImportError as e:
        print(f"✗ Keras import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    return True

def test_model_file():
    """Check if model file exists"""
    import os
    print("\nTesting model file...")
    
    model_path = "./model/emo.h5"
    if os.path.exists(model_path):
        print(f"✓ Model file found: {model_path}")
        
        # Check file size
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  File size: {size_mb:.2f} MB")
        return True
    else:
        print(f"✗ Model file not found: {model_path}")
        return False

def test_camera():
    """Test camera access"""
    print("\nTesting camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            print("✓ Camera accessible")
            cap.release()
            return True
        else:
            print("✗ Camera not accessible")
            return False
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Emotion Detection System - Installation Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test model file
    model_ok = test_model_file()
    
    # Test camera
    camera_ok = test_camera()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print("=" * 50)
    print(f"Package Imports: {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"Model File:      {'✓ PASS' if model_ok else '✗ FAIL'}")
    print(f"Camera Access:   {'✓ PASS' if camera_ok else '✗ FAIL'}")
    
    if imports_ok and model_ok and camera_ok:
        print("\n🎉 All tests passed! You're ready to run the emotion detection system.")
        print("Run: python EmotionDetection.py")
    else:
        print("\n⚠️  Some tests failed. Please check the installation.")
        if not imports_ok:
            print("- Install required packages: pip install -r requirements.txt")
        if not model_ok:
            print("- Ensure model file 'emo.h5' is in the 'model/' directory")
        if not camera_ok:
            print("- Check camera connection and permissions")

if __name__ == "__main__":
    main()