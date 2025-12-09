# Real-Time Emotion Detection System

## 🎯 Project Overview

This project implements a **Real-Time Emotion Detection System** using Deep Learning techniques for the NNDL (Neural Networks and Deep Learning) course. The system uses computer vision to detect human faces in real-time video streams and classifies emotions using a pre-trained Convolutional Neural Network (CNN).

## 📊 Dataset

**FER-2013 (Facial Expression Recognition 2013)**
- **Source**: Kaggle Competition Dataset
- **Size**: ~35,000 grayscale images (48x48 pixels)
- **Classes**: 7 emotion categories
- **Distribution**: Training (~28k), Validation (~3.5k), Test (~3.5k)
- **Format**: CSV with pixel values and emotion labels

### Emotion Classes
1. **Angry** 😠
2. **Disgust** 🤢  
3. **Fear** 😨
4. **Happy** 😊
5. **Sad** 😢
6. **Surprise** 😲
7. **Neutral** 😐

## 🏗️ System Architecture

### Model Architecture
- **Input**: 48x48 grayscale images
- **Model Type**: Convolutional Neural Network (CNN)
- **Output**: 7-class probability distribution
- **Framework**: TensorFlow/Keras

### Pipeline Components
1. **Face Detection**: Haar Cascade Classifier
2. **Preprocessing**: Grayscale conversion, resizing, normalization
3. **Emotion Classification**: CNN model inference
4. **Visualization**: Real-time bounding boxes and emotion labels

## 🛠️ Technical Implementation

### Core Technologies
- **Python 3.x**: Primary programming language
- **OpenCV**: Computer vision operations and webcam interface
- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations

### Key Features
- ✅ Real-time face detection
- ✅ Multi-face emotion recognition
- ✅ Confidence score display
- ✅ Live video stream processing
- ✅ Optimized preprocessing pipeline

## 📁 Project Structure

```
Emotion_Detection/
├── EmotionDetection.py     # Main application script
├── requirements.txt        # Python dependencies
├── model/
│   └── emo.h5             # Pre-trained CNN model
└── README.md              # Project documentation
```

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.7 or higher
- Webcam or camera device
- Git (for cloning)

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd Emotion_Detection
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv emotion_env
   
   # On Windows:
   emotion_env\Scripts\activate
   
   # On macOS/Linux:
   source emotion_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import cv2, tensorflow; print('Installation successful!')"
   ```

## 🚀 Usage Instructions

### Running the Application

1. **Start the emotion detection system**
   ```bash
   python EmotionDetection.py
   ```

2. **Using the system**
   - Position yourself in front of the camera
   - Ensure good lighting conditions
   - The system will automatically detect faces and classify emotions
   - Emotion labels and confidence scores will appear above detected faces

3. **Exit the application**
   - Press `'q'` key to quit the application

### Expected Output
- Real-time video feed with:
  - Green bounding boxes around detected faces
  - Emotion labels (e.g., "Happy: 87.45%")
  - Confidence percentages for predictions

## 🔧 Algorithm Details

### Face Detection Pipeline
```python
# Haar Cascade Parameters
scaleFactor = 1.3      # Image pyramid scaling
minNeighbors = 5       # Minimum neighbor rectangles
```

### Preprocessing Steps
1. **Color Space Conversion**: BGR → Grayscale
2. **Face Extraction**: ROI extraction using bounding box
3. **Resizing**: Standardize to 48x48 pixels
4. **Normalization**: Pixel values scaled to [0,1] range
5. **Tensor Reshaping**: Add batch and channel dimensions

### Model Inference
- **Input Shape**: (1, 48, 48, 1)
- **Output Shape**: (1, 7) - probability distribution
- **Prediction**: Argmax for final class selection

## 📈 Performance Metrics

### Model Performance (Estimated)
- **Training Accuracy**: ~65-70%
- **Validation Accuracy**: ~60-65%
- **Real-time FPS**: 15-30 FPS (depending on hardware)

### System Requirements
- **Minimum RAM**: 4GB
- **Recommended RAM**: 8GB+
- **CPU**: Multi-core processor recommended
- **GPU**: Optional (CUDA-compatible for faster inference)

## 🔍 Troubleshooting

### Common Issues & Solutions

1. **Camera not detected**
   ```python
   # Try different camera indices
   cap = cv2.VideoCapture(1)  # or 2, 3, etc.
   ```

2. **Model loading error**
   - Ensure `emo.h5` file is in the correct `model/` directory
   - Verify file integrity and compatibility

3. **Poor emotion detection**
   - Improve lighting conditions
   - Ensure face is clearly visible
   - Maintain appropriate distance from camera

4. **Performance issues**
   - Close unnecessary applications
   - Reduce video resolution if needed
   - Consider GPU acceleration

## 🚀 Future Enhancements

### Potential Improvements
- [ ] **Model Optimization**: Quantization for mobile deployment
- [ ] **Multi-modal Analysis**: Audio-visual emotion recognition
- [ ] **Data Augmentation**: Improved training with synthetic data
- [ ] **Edge Computing**: Deployment on embedded systems
- [ ] **Real-time Analytics**: Emotion tracking and analytics dashboard

### Advanced Features
- [ ] **Age and Gender Detection**: Multi-task learning
- [ ] **Attention Mechanisms**: Focus on facial regions
- [ ] **3D Face Analysis**: Depth-based emotion recognition
- [ ] **Cloud Integration**: Scalable emotion analytics platform

## 📚 Technical References

### Research Papers
1. "Challenges in Representation Learning: A report on three machine learning contests" - Goodfellow et al.
2. "Deep Learning for Facial Expression Recognition: A Survey" - Li et al.
3. "FER-2013 Dataset Analysis and Emotion Recognition" - Various studies

### Frameworks & Libraries
- [TensorFlow Documentation](https://tensorflow.org/docs)
- [OpenCV Python Tutorials](https://opencv-python-tutroals.readthedocs.io)
- [Keras API Reference](https://keras.io/api/)

## 👥 Contributing

### Development Guidelines
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings for functions
- Include unit tests for new features
- Update documentation accordingly

## 📄 License

This project is developed for educational purposes as part of the NNDL course curriculum. Please respect academic integrity guidelines when using this code.

## 📞 Support & Contact

For technical questions or collaboration opportunities:
- **Course**: Neural Networks and Deep Learning (NNDL)
- **Institution**: [Your Institution Name]
- **Academic Year**: [Your Academic Year]

---

**⭐ If you found this project helpful, please star the repository!**

---

## 🔖 Keywords
`emotion-recognition` `deep-learning` `computer-vision` `fer2013` `tensorflow` `keras` `opencv` `real-time` `face-detection` `neural-networks` `machine-learning` `python` `nndl-project`