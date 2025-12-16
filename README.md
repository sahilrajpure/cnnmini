# ShadowClear - AI Shadow Detection and Removal

An interactive web application built with Streamlit and TensorFlow to detect and remove shadows from document images using deep learning (U-Net architecture).

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-latest-red.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

ShadowClear is an AI-powered document enhancement tool that automatically detects and removes shadows from scanned documents and images. The application uses a U-Net convolutional neural network trained on the ISTD (Image Shadow Triplets Dataset) to achieve 98%+ accuracy in shadow detection and removal.

## Features

- **3 Interactive Sections**: Introduction, Dataset Information, and Model Implementation
- **Real-time Shadow Detection**: Upload images and get instant shadow detection results
- **Comprehensive Visualizations**: Shadow boundaries, binary masks, and shadow-free outputs
- **Multiple Download Options**: Download detected shadows, masks, and final cleaned images
- **Educational Content**: Detailed explanations of CNN, U-Net architecture, and shadow detection concepts
- **Dataset Explorer**: View sample images from the ISTD dataset

## Technologies Used

- **Streamlit** - Interactive web framework
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Image processing
- **NumPy** - Numerical computing
- **Pillow** - Image manipulation

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- GPU recommended for faster processing (optional)

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/sahilrajpure/shadowclear.git
cd shadowclear
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure required files are present:
   - `shadow_removal_final.h5` - Trained U-Net model
   - `cnnarchitect.png` - CNN architecture diagram
   - `unetarchitect.png` - U-Net architecture diagram
   - `dataset_samples/` - Sample dataset images (optional)

## Usage

Run the application:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure

```
shadowclear/
│
├── app.py                      # Main application file
├── shadow_removal_final.h5     # Trained U-Net model
├── cnnarchitect.png           # CNN architecture visualization
├── unetarchitect.png          # U-Net architecture visualization
├── requirements.txt           # Python dependencies
├── dataset_samples/           # Sample ISTD images (optional)
└── README.md                  # Documentation
```

## Model Information

- **Architecture**: U-Net (Encoder-Decoder with Skip Connections)
- **Dataset**: ISTD - Image Shadow Triplets Dataset
- **Training Images**: 500 triplets
- **Testing Images**: 540 triplets
- **Input Size**: 256×256 pixels (RGB)
- **Training Epochs**: 50
- **Accuracy**: 98%+

## Model Performance

The U-Net model achieves exceptional performance on the ISTD test set:

- **Shadow Detection Accuracy**: 98%+
- **Image Processing**: Real-time shadow removal
- **Output Quality**: High-quality shadow-free images suitable for OCR
- **Generalization**: Works across various document types and lighting conditions

## Application Sections

### 1. Introduction
- CNN fundamentals and architecture types
- U-Net architecture explanation
- Key terminology and concepts
- Advantages and disadvantages
- Real-world applications

### 2. Dataset Information
- ISTD dataset overview and statistics
- Dataset composition and structure
- Processing pipeline flowchart
- Sample image viewer
- Dataset download link

### 3. Model Implementation
- Model loading and status
- Image upload interface
- Real-time processing pipeline
- Visual results (detection, mask, output)
- Before/after comparisons
- Multi-format downloads

## Supported File Formats

**Input Formats:**
- JPG / JPEG
- PNG
- BMP

**Output Format:**
- PNG (lossless, high quality)

## Applications

- **Document Processing**: Scanning, digitization, and archiving
- **OCR Enhancement**: Improved text recognition accuracy
- **Cultural Heritage**: Historical document restoration
- **Business**: Invoice processing, contract digitization
- **Education**: Textbook and note scanning
- **Photography**: General photo enhancement

## Dataset

The ISTD dataset can be downloaded from Google Drive:
[Download ISTD Dataset](https://drive.google.com/drive/folders/10KezAOQtKbfw3ybzm4aaor03Pi9_G_Z4?usp=sharing)


### Introduction Page
[![shadowclearhome.png](https://i.postimg.cc/ZY78Nqhg/shadowclearhome.png)](https://postimg.cc/xqmNDn8y)
*CNN and U-Net architecture explanations with detailed visualizations*

### Dataset Information
[![shadowcleardataset.png](https://i.postimg.cc/HW40kt6K/shadowcleardataset.png)](https://postimg.cc/mhk1d7fN)
*ISTD dataset statistics, flowcharts, and sample images with download option*

### Model Implementation & Results
[![shadowclearimplementation.png](https://i.postimg.cc/XJnffGJz/shadowclearimplementation.png)](https://postimg.cc/zLPH5fTC)
*Real-time shadow detection and removal with before/after comparison and downloadable results*

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add NewFeature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## Future Enhancements

- Support for batch processing multiple images
- Additional shadow removal algorithms comparison
- Video shadow removal capability
- API endpoint for integration
- Mobile app version
- Cloud deployment option
- Custom model training interface

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Sahil Rajpure**
- GitHub: [@sahilrajpure](https://github.com/sahilrajpure)
- LinkedIn: [Sahil Rajpure](https://www.linkedin.com/in/sahil-rajpure-b2891924b/)
- Email: rajpuresahilcs222335@gmail.com

## Acknowledgments

- ISTD Dataset creators for providing high-quality training data
- U-Net architecture (Ronneberger et al., 2015)
- Streamlit community for excellent documentation
- TensorFlow/Keras for deep learning framework

## Citation

If you use this project in your research, please cite:

```
@software{shadowclear2025,
  author = {Sahil Rajpure},
  title = {ShadowClear: AI-Powered Shadow Detection and Removal},
  year = {2025},
  url = {https://github.com/sahilrajpure/shadowclear}
}
```

## Contact

For questions or feedback:
- Email: rajpuresahilcs222335@gmail.com
- LinkedIn: [Sahil Rajpure](https://www.linkedin.com/in/sahil-rajpure-b2891924b/)
- GitHub Issues: [Project Issues](https://github.com/sahilrajpure/shadowclear/issues)

---

⭐ If you find this project helpful, please give it a star!