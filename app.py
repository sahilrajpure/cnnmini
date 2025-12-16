import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import gdown
import io
import os

st.set_page_config(
    page_title="ShadowClear - AI Shadow Removal",
    page_icon="🌓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
        padding: 1rem;
    }
    
    .stButton>button {
        background-color: #0066cc;
        color: white;
        font-size: clamp(14px, 3vw, 16px);
        font-weight: bold;
        border-radius: 8px;
        padding: clamp(10px, 2vh, 12px) clamp(20px, 4vw, 28px);
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0052a3;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,102,204,0.3);
    }
    
    h1 {
        color: #1a1a1a;
        text-align: center;
        font-weight: 700;
        padding: clamp(15px, 3vh, 20px) 0;
        font-size: clamp(24px, 5vw, 36px);
    }
    h2 {
        color: #333333;
        font-weight: 600;
        border-bottom: 3px solid #0066cc;
        padding-bottom: 10px;
        font-size: clamp(20px, 4vw, 28px);
        margin-top: 1.5rem;
    }
    h3 {
        color: #555555;
        font-weight: 500;
        font-size: clamp(16px, 3.5vw, 22px);
    }
    h4 {
        font-size: clamp(14px, 3vw, 18px);
    }
    
    .info-box, .success-box, .warning-box {
        padding: clamp(12px, 2.5vw, 15px);
        margin: clamp(10px, 2vh, 15px) 0;
        border-radius: 8px;
        font-size: clamp(13px, 2.5vw, 15px);
        line-height: 1.6;
    }
    .info-box {
        background-color: #e7f3ff;
        border-left: 5px solid #0066cc;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    
    .metric-card {
        background-color: #fafafa;
        padding: clamp(15px, 3vw, 20px);
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        margin-bottom: 1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    .section-card {
        background-color: #fafafa;
        padding: clamp(15px, 3.5vw, 25px);
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: clamp(15px, 2.5vh, 20px) 0;
    }
    .section-card ul, .section-card p {
        font-size: clamp(13px, 2.5vw, 15px);
        line-height: 1.7;
    }
    
    @media (max-width: 768px) {
        .row-widget.stHorizontalBlock {
            flex-direction: column;
        }
        .metric-card, .section-card {
            margin-bottom: 1rem;
        }
        h2 {
            font-size: 22px;
        }
    }
    
    @media (min-width: 769px) and (max-width: 1024px) {
        .main {
            padding: 1.5rem;
        }
    }
    
    .stImage {
        border-radius: 8px;
        overflow: hidden;
    }
    
    .element-container {
        margin-bottom: 0.5rem;
    }
    
    @media (max-width: 768px) {
        section[data-testid="stSidebar"] {
            width: 100% !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        file_id = "1wzglObjQwryoz4Dry6ydi5Tg0NJAmBEo"
        url = f"https://drive.google.com/uc?id={file_id}"
        model_path = "shadow_removal_final.h5"

        if not os.path.exists(model_path):
            gdown.download(url, model_path, quiet=False)

        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )
        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Ensure the Drive file is shared as 'Anyone with the link'.")
        return None


def preprocess_image(image, img_size=256):
    img = np.array(image)
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    original_img = img.copy()
    img_resized = cv2.resize(img, (img_size, img_size))
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    return img_normalized, original_img

def draw_red_boundary(image, mask, thickness=3):
    img_with_boundary = (image * 255).astype(np.uint8).copy()
    mask_binary = (mask.squeeze() > 0.5).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_with_boundary, contours, -1, (255, 0, 0), thickness)
    
    return img_with_boundary

def resize_to_original(processed_img, original_shape):
    processed_uint8 = (processed_img * 255).astype(np.uint8)
    resized = cv2.resize(processed_uint8, (original_shape[1], original_shape[0]))
    return resized

def convert_to_downloadable(image):
    img_pil = Image.fromarray(image)
    buf = io.BytesIO()
    img_pil.save(buf, format='PNG')
    byte_im = buf.getvalue()
    return byte_im

# Sidebar Navigation
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #0066cc;'>🌓 ShadowClear</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; font-size: 12px;'>Shadow Detection and Removal</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio(
        "Navigate to:",
        ["Introduction", "Dataset Information", "Model Implementation"],
        label_visibility="visible"
    )
    
    st.markdown("---")
    
    st.markdown("### Quick Info")
    st.markdown("""
    - **Architecture**: U-Net (CNN)
    - **Dataset**: ISTD
    - **Accuracy**: 98%+
    - **Input Size**: 256×256
    - **Output**: Shadow-free Image
    """)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    <div style='font-size: 12px; color: #666;'>
    ShadowClear uses deep learning to automatically detect and remove shadows from document images.
    </div>
    """, unsafe_allow_html=True)

# PAGE 1: INTRODUCTION
if page == "Introduction":
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 40px; border-radius: 10px; margin-bottom: 30px;'>
            <h1 style='color: white; margin: 0;'>🌓 ShadowClear</h1>
            <p style='color: white; text-align: center; font-size: 18px; margin-top: 10px;'>
                (Shadow Detection and Removal)
            </p>
            <p style='color: white; text-align: center; font-size: 14px; margin-top: 5px;'>
                AI-Powered Document Enhancement using Deep Learning
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # CNN Definition
    st.markdown("## Convolutional Neural Networks (CNN)")
    st.markdown("""
    <div class='section-card'>
    <h3>What is CNN?</h3>
    <p style='font-size: 16px; line-height: 1.8;'>
    A Convolutional Neural Network (CNN) is a specialized deep learning architecture designed for processing 
    structured grid data like images. CNNs automatically learn spatial hierarchies of features through 
    backpropagation, making them highly effective for computer vision tasks.
    </p>
    
    <h3>Key Characteristics:</h3>
    <ul style='font-size: 15px; line-height: 1.8;'>
        <li><strong>Local Connectivity:</strong> Neurons connect to local regions of input</li>
        <li><strong>Parameter Sharing:</strong> Same filter applied across entire image</li>
        <li><strong>Translation Invariance:</strong> Can detect features regardless of position</li>
        <li><strong>Hierarchical Learning:</strong> Builds from simple to complex features</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Types of CNN
    st.markdown("## Types of CNN Architectures")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='section-card'>
        <h3>Classification CNNs</h3>
        <ul style='font-size: 14px; line-height: 1.8;'>
            <li><strong>LeNet:</strong> Early CNN for digit recognition</li>
            <li><strong>AlexNet:</strong> Deep CNN with ReLU activation</li>
            <li><strong>VGGNet:</strong> Very deep with small filters</li>
            <li><strong>ResNet:</strong> Residual connections for depth</li>
            <li><strong>Inception:</strong> Multi-scale feature extraction</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='section-card'>
        <h3>Segmentation CNNs</h3>
        <ul style='font-size: 14px; line-height: 1.8;'>
            <li><strong>FCN:</strong> Fully Convolutional Networks</li>
            <li><strong>U-Net:</strong> Encoder-decoder with skip connections</li>
            <li><strong>SegNet:</strong> Symmetric encoder-decoder</li>
            <li><strong>DeepLab:</strong> Atrous convolution</li>
            <li><strong>Mask R-CNN:</strong> Instance segmentation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # CNN Architecture
    st.markdown("## CNN Architecture Components")
    
    st.markdown("""
    <div class='section-card'>
    <h3>Typical CNN Architecture Flow:</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h4>Input Layer</h4>
            <p style='font-size: 13px;'>Raw image pixels</p>
            <p style='font-size: 12px; color: #666;'>H × W × C</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h4>Convolution</h4>
            <p style='font-size: 13px;'>Feature extraction</p>
            <p style='font-size: 12px; color: #666;'>Filters/Kernels</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h4>Pooling</h4>
            <p style='font-size: 13px;'>Downsampling</p>
            <p style='font-size: 12px; color: #666;'>Max/Avg Pool</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='metric-card'>
            <h4>Output</h4>
            <p style='font-size: 13px;'>Prediction</p>
            <p style='font-size: 12px; color: #666;'>Classification/Mask</p>
        </div>
        """, unsafe_allow_html=True)
    
    # CNN Architecture Diagram
    st.markdown("### CNN Architecture Visualization")
    st.markdown("""
    <div class='info-box'>
    <strong>Example CNN Architecture:</strong> The diagram below shows a typical CNN structure for image processing.
    </div>
    """, unsafe_allow_html=True)
    
    st.image("cnnarchitect.png", 
             caption="Typical CNN Architecture: Input → Conv → Pool → Conv → Pool → Flatten → FC → Output")
    
    # U-Net Definition
    st.markdown("## U-Net Architecture (CNN-based)")
    
    st.markdown("""
    <div class='section-card'>
    <h3>What is U-Net?</h3>
    <p style='font-size: 16px; line-height: 1.8;'>
    U-Net is a specialized CNN architecture designed for image segmentation tasks. It features a symmetric 
    encoder-decoder structure with skip connections that form a "U" shape. Originally developed for biomedical 
    image segmentation, U-Net has become the gold standard for pixel-wise prediction tasks.
    </p>
    
    <h3>Architecture Components:</h3>
    <ul style='font-size: 15px; line-height: 1.8;'>
        <li><strong>Contracting Path (Encoder):</strong> Captures context through convolutions and pooling</li>
        <li><strong>Bottleneck:</strong> Deepest layer with maximum feature compression</li>
        <li><strong>Expanding Path (Decoder):</strong> Enables precise localization through upsampling</li>
        <li><strong>Skip Connections:</strong> Concatenates encoder features with decoder for better spatial accuracy</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # U-Net Architecture Diagram
    st.markdown("### U-Net Architecture Visualization")
    st.image("unetarchitect.png", 
             caption="U-Net Architecture: Encoder (left) → Bottleneck (bottom) → Decoder (right) with Skip Connections")
    
    # Key Terms
    st.markdown("## Key Terms in Shadow Detection")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='section-card'>
        <h3>CNN Terms</h3>
        <ul style='font-size: 13px; line-height: 1.6;'>
            <li><strong>Convolution:</strong> Feature extraction operation</li>
            <li><strong>Kernel/Filter:</strong> Weight matrix for convolution</li>
            <li><strong>Stride:</strong> Step size of filter movement</li>
            <li><strong>Padding:</strong> Border pixels added to input</li>
            <li><strong>Activation:</strong> Non-linear transformation (ReLU)</li>
            <li><strong>Pooling:</strong> Downsampling operation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='section-card'>
        <h3>U-Net Terms</h3>
        <ul style='font-size: 13px; line-height: 1.6;'>
            <li><strong>Encoder:</strong> Feature extraction path</li>
            <li><strong>Decoder:</strong> Reconstruction path</li>
            <li><strong>Skip Connection:</strong> Direct feature transfer</li>
            <li><strong>Bottleneck:</strong> Deepest compressed layer</li>
            <li><strong>Upsampling:</strong> Increase spatial resolution</li>
            <li><strong>Concatenation:</strong> Feature merging</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='section-card'>
        <h3>Shadow Terms</h3>
        <ul style='font-size: 13px; line-height: 1.6;'>
            <li><strong>Shadow Mask:</strong> Binary segmentation map</li>
            <li><strong>Illumination:</strong> Light intensity distribution</li>
            <li><strong>LAB Color Space:</strong> Lightness + color channels</li>
            <li><strong>Luminance:</strong> Brightness component</li>
            <li><strong>Shadow Boundary:</strong> Edge between regions</li>
            <li><strong>Correction Factor:</strong> Brightness adjustment</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Advantages & Disadvantages
    st.markdown("## U-Net for Shadow Detection: Pros & Cons")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='section-card' style='border-left: 5px solid #28a745;'>
        <h3 style='color: #28a745;'>Advantages</h3>
        <ul style='font-size: 14px; line-height: 1.8;'>
            <li><strong>High Accuracy:</strong> Precise pixel-level segmentation of shadow regions</li>
            <li><strong>Skip Connections:</strong> Preserves fine spatial details and boundaries</li>
            <li><strong>End-to-End Learning:</strong> No manual feature engineering required</li>
            <li><strong>Robust to Variations:</strong> Handles different shadow types and intensities</li>
            <li><strong>Efficient:</strong> Works well even with limited training data</li>
            <li><strong>Generalizable:</strong> Transfers to various document types</li>
            <li><strong>Fast Inference:</strong> Real-time processing capability</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='section-card' style='border-left: 5px solid #dc3545;'>
        <h3 style='color: #dc3545;'>Disadvantages</h3>
        <ul style='font-size: 14px; line-height: 1.8;'>
            <li><strong>Training Data:</strong> Requires paired shadow/shadow-free images</li>
            <li><strong>Computational Cost:</strong> Needs GPU for training and inference</li>
            <li><strong>Memory Usage:</strong> Large model with many parameters</li>
            <li><strong>Fixed Input Size:</strong> Requires image resizing (256×256)</li>
            <li><strong>Complex Shadows:</strong> May struggle with extreme lighting conditions</li>
            <li><strong>Model Size:</strong> Large file size for deployment</li>
            <li><strong>Overfitting Risk:</strong> Can overfit on small datasets</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Applications
    st.markdown("## Applications of Shadow Detection & Removal")
    
    st.markdown("""
    <div class='section-card'>
    <h3>Real-World Use Cases:</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h4>Document Processing</h4>
            <ul style='font-size: 13px; text-align: left; line-height: 1.6;'>
                <li>Document scanning & digitization</li>
                <li>Archive preservation</li>
                <li>Professional document editing</li>
                <li>Book scanning</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h4>OCR Enhancement</h4>
            <ul style='font-size: 13px; text-align: left; line-height: 1.6;'>
                <li>Improved text recognition</li>
                <li>Better character detection</li>
                <li>Enhanced readability</li>
                <li>Automated data extraction</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h4>Photography</h4>
            <ul style='font-size: 13px; text-align: left; line-height: 1.6;'>
                <li>Photo enhancement</li>
                <li>Portrait editing</li>
                <li>Real estate photography</li>
                <li>Product photography</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h4>Cultural Heritage</h4>
            <ul style='font-size: 13px; text-align: left; line-height: 1.6;'>
                <li>Historical document restoration</li>
                <li>Museum artifact digitization</li>
                <li>Ancient manuscript preservation</li>
                <li>Art conservation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h4>Business Applications</h4>
            <ul style='font-size: 13px; text-align: left; line-height: 1.6;'>
                <li>Invoice processing</li>
                <li>Contract digitization</li>
                <li>Form scanning</li>
                <li>Receipt management</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h4>Education</h4>
            <ul style='font-size: 13px; text-align: left; line-height: 1.6;'>
                <li>Textbook digitization</li>
                <li>Note scanning</li>
                <li>Exam paper processing</li>
                <li>Research paper archiving</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# PAGE 2: DATASET INFORMATION
elif page == "Dataset Information":
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 40px; border-radius: 10px; margin-bottom: 30px;'>
            <h1 style='color: white; margin: 0;'>Dataset Information</h1>
            <p style='color: white; text-align: center; font-size: 18px; margin-top: 10px;'>
                ISTD - Image Shadow Triplets Dataset
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Dataset Overview
    st.markdown("## ISTD Dataset Overview")
    st.markdown("""
    <div class='section-card'>
    <p style='font-size: 16px; line-height: 1.8;'>
    The Image Shadow Triplets Dataset (ISTD) is a comprehensive dataset specifically designed for shadow 
    detection and removal tasks. It contains high-quality image triplets consisting of shadow images, 
    shadow masks, and shadow-free ground truth images.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset Statistics
    st.markdown("## Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #0066cc; margin: 0;'>Total Images</h3>
            <h2 style='color: #333; margin: 10px 0;'>1,330+</h2>
            <p style='font-size: 12px; color: #666;'>Image Triplets</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #28a745; margin: 0;'>Training Set</h3>
            <h2 style='color: #333; margin: 10px 0;'>500</h2>
            <p style='font-size: 12px; color: #666;'>Images (37.6%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #ffc107; margin: 0;'>Testing Set</h3>
            <h2 style='color: #333; margin: 10px 0;'>540</h2>
            <p style='font-size: 12px; color: #666;'>Images (40.6%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #dc3545; margin: 0;'>Input Size</h3>
            <h2 style='color: #333; margin: 10px 0;'>256×256</h2>
            <p style='font-size: 12px; color: #666;'>Pixels (Resized)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Dataset Composition
    st.markdown("## Dataset Flow & Composition")
    
    st.markdown("""
    <div class='section-card'>
    <h3>Each Dataset Sample Contains Three Images:</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; text-align: center;'>
            <h4 style='color: white; margin: 0;'>Image A</h4>
            <p style='color: white; font-size: 14px; margin: 10px 0;'>Shadow Image</p>
            <p style='color: white; font-size: 12px;'>(Input)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                    padding: 20px; border-radius: 10px; text-align: center;'>
            <h4 style='color: white; margin: 0;'>Image B</h4>
            <p style='color: white; font-size: 14px; margin: 10px 0;'>Shadow-Free Image</p>
            <p style='color: white; font-size: 12px;'>(Target Output)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%); 
                    padding: 20px; border-radius: 10px; text-align: center;'>
            <h4 style='color: white; margin: 0;'>Image C</h4>
            <p style='color: white; font-size: 14px; margin: 10px 0;'>Shadow Mask</p>
            <p style='color: white; font-size: 12px;'>(Ground Truth)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Dataset Flowchart
    st.markdown("## Dataset Processing Flowchart")
    
    st.markdown("""
    <div class='section-card'>
    <h3>Training & Testing Pipeline:</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
            <h4 style='text-align: center; color: #0066cc;'>📥 INPUT STAGE</h4>
            <div style='margin: 20px 0;'>
                <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;'>
                    <p style='text-align: center; font-weight: bold; margin: 5px 0;'>Raw Dataset</p>
                    <p style='text-align: center; font-size: 12px; color: #666;'>ISTD Image Triplets</p>
                </div>
                <div style='text-align: center; margin: 10px 0;'>⬇️</div>
                <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;'>
                    <p style='text-align: center; font-weight: bold; margin: 5px 0;'>Preprocessing</p>
                    <p style='text-align: center; font-size: 12px; color: #666;'>Resize to 256×256</p>
                </div>
                <div style='text-align: center; margin: 10px 0;'>⬇️</div>
                <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;'>
                    <p style='text-align: center; font-weight: bold; margin: 5px 0;'>Normalization</p>
                    <p style='text-align: center; font-size: 12px; color: #666;'>Values: [0, 1]</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
            <h4 style='text-align: center; color: #28a745;'>🔄 PROCESSING STAGE</h4>
            <div style='margin: 20px 0;'>
                <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;'>
                    <p style='text-align: center; font-weight: bold; margin: 5px 0;'>Data Augmentation</p>
                    <p style='text-align: center; font-size: 12px; color: #666;'>Rotation, Flip, Crop</p>
                </div>
                <div style='text-align: center; margin: 10px 0;'>⬇️</div>
                <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;'>
                    <p style='text-align: center; font-weight: bold; margin: 5px 0;'>U-Net Training</p>
                    <p style='text-align: center; font-size: 12px; color: #666;'>50 Epochs</p>
                </div>
                <div style='text-align: center; margin: 10px 0;'>⬇️</div>
                <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;'>
                    <p style='text-align: center; font-weight: bold; margin: 5px 0;'>Validation</p>
                    <p style='text-align: center; font-size: 12px; color: #666;'>Loss & Accuracy</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
            <h4 style='text-align: center; color: #ffc107;'>📤 OUTPUT STAGE</h4>
            <div style='margin: 20px 0;'>
                <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;'>
                    <p style='text-align: center; font-weight: bold; margin: 5px 0;'>Trained Model</p>
                    <p style='text-align: center; font-size: 12px; color: #666;'>shadow_removal_final.h5</p>
                </div>
                <div style='text-align: center; margin: 10px 0;'>⬇️</div>
                <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;'>
                    <p style='text-align: center; font-weight: bold; margin: 5px 0;'>Testing</p>
                    <p style='text-align: center; font-size: 12px; color: #666;'>540 Test Images</p>
                </div>
                <div style='text-align: center; margin: 10px 0;'>⬇️</div>
                <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;'>
                    <p style='text-align: center; font-weight: bold; margin: 5px 0;'>Deployment</p>
                    <p style='text-align: center; font-size: 12px; color: #666;'>98%+ Accuracy</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Dataset Details
    st.markdown("## Dataset Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='section-card'>
        <h3>Image Specifications</h3>
        <ul style='font-size: 15px; line-height: 1.8;'>
            <li><strong>Format:</strong> PNG, JPG</li>
            <li><strong>Original Resolution:</strong> Variable (480-640px typical)</li>
            <li><strong>Training Resolution:</strong> Resized to 256×256</li>
            <li><strong>Color Space:</strong> RGB (3 channels)</li>
            <li><strong>Bit Depth:</strong> 8-bit per channel</li>
            <li><strong>File Size:</strong> 50-200 KB per image</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='section-card'>
        <h3>Dataset Characteristics</h3>
        <ul style='font-size: 15px; line-height: 1.8;'>
            <li><strong>Scene Types:</strong> Indoor & Outdoor</li>
            <li><strong>Shadow Types:</strong> Hard & Soft shadows</li>
            <li><strong>Lighting:</strong> Natural & Artificial</li>
            <li><strong>Shadow Coverage:</strong> 10-60% of image area</li>
            <li><strong>Object Types:</strong> Documents, papers, books</li>
            <li><strong>Backgrounds:</strong> Various surfaces & textures</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # View Top 3 Images Section
    st.markdown("## View 3 Dataset Samples")
    
    st.markdown("""
    <div class='info-box'>
    <strong>Sample Images:</strong> Below are representative samples from the ISTD dataset showing shadow images, 
    their corresponding masks, and shadow-free ground truth images.
    </div>
    """, unsafe_allow_html=True)
    
    # Check if sample images exist
    sample_dir = "dataset_samples"
    if os.path.exists(sample_dir):
        sample_files = sorted([f for f in os.listdir(sample_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        if len(sample_files) >= 9:
            for i in range(3):
                st.markdown(f"### Sample {i+1}")
                
                col1, col2, col3 = st.columns(3)
                
                # Try to load triplet images (assuming naming convention)
                base_name = sample_files[i*3].split('_')[0]
                
                with col1:
                    shadow_img_path = os.path.join(sample_dir, f"{base_name}_shadow.png")
                    if os.path.exists(shadow_img_path):
                        st.image(shadow_img_path, caption=f"Shadow Image {i+1}", use_column_width=True)
                    else:
                        st.image(os.path.join(sample_dir, sample_files[i]), 
                               caption=f"Sample Image {i+1}", use_column_width=True)
                
                with col2:
                    mask_img_path = os.path.join(sample_dir, f"{base_name}_mask.png")
                    if os.path.exists(mask_img_path):
                        st.image(mask_img_path, caption=f"Shadow Mask {i+1}", use_column_width=True)
                    else:
                        st.info("Mask not available")
                
                with col3:
                    clean_img_path = os.path.join(sample_dir, f"{base_name}_clean.png")
                    if os.path.exists(clean_img_path):
                        st.image(clean_img_path, caption=f"Shadow-Free {i+1}", use_column_width=True)
                    else:
                        st.info("Clean image not available")
                
                st.markdown("---")
        else:
            st.warning("Not enough sample images found. Please add at least 3 sample images to 'dataset_samples' folder.")
    
    
    # Dataset Download Section
    st.markdown("## Dataset Download")
    
    st.markdown("""
    <div class='section-card'>
    <h3>ISTD Dataset</h3>
    <p style='font-size: 15px; line-height: 1.8;'>
    If you want to use the ISTD (Image Shadow Triplets Dataset) for this project,
    you can download it directly from Google Drive.
    </p>
    <p style='font-size: 15px; line-height: 1.8;'>
    The dataset contains shadow images, corresponding shadow masks, and shadow-free
    ground truth images.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(
        """
        <a href="https://drive.google.com/drive/folders/10KezAOQtKbfw3ybzm4aaor03Pi9_G_Z4?usp=sharing"
        target="_blank">
        <button style="
            background-color:#0066cc;
            color:white;
            padding:12px 24px;
            font-size:16px;
            border:none;
            border-radius:6px;
            cursor:pointer;">
            Download ISTD Dataset (ZIP)
        </button>
        </a>
        """,unsafe_allow_html=True)
    
    st.markdown("""
<div class='info-box'>
<strong>Note:</strong> Open the link and use <b>Download → ZIP</b> to download the full dataset.
</div>
""", unsafe_allow_html=True)


# PAGE 3: MODEL IMPLEMENTATION
elif page == "Model Implementation":
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 40px; border-radius: 10px; margin-bottom: 30px;'>
            <h1 style='color: white; margin: 0;'>Model Implementation</h1>
            <p style='color: white; text-align: center; font-size: 18px; margin-top: 10px;'>
                ShadowClear - Upload and Process Your Documents
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load Model
    model = load_model()
    
    if model is None:
        st.markdown("""
        <div class='warning-box'>
        <strong>Model Not Found!</strong><br>
        Please ensure that 'shadow_removal_final.h5' is in the same directory as this application.
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    st.markdown("""
    <div class='success-box'>
    <strong>Model Status:</strong> Successfully Loaded | U-Net Architecture Ready for Processing
    </div>
    """, unsafe_allow_html=True)
    
    # Model Information
    st.markdown("## Model Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h4>Architecture</h4>
            <p style='font-size: 16px; font-weight: bold; color: #0066cc;'>U-Net</p>
            <p style='font-size: 12px; color: #666;'>Encoder-Decoder</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h4>Input Size</h4>
            <p style='font-size: 16px; font-weight: bold; color: #0066cc;'>256×256</p>
            <p style='font-size: 12px; color: #666;'>RGB Channels</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h4>Training Epochs</h4>
            <p style='font-size: 16px; font-weight: bold; color: #0066cc;'>50</p>
            <p style='font-size: 12px; color: #666;'>Iterations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='metric-card'>
            <h4>Accuracy</h4>
            <p style='font-size: 16px; font-weight: bold; color: #28a745;'>98%+</p>
            <p style='font-size: 12px; color: #666;'>Test Set</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Upload Section
    st.markdown("## Upload Shadow Document")
    
    uploaded_file = st.file_uploader(
        "Choose an image file with shadows",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a document image containing shadows for processing"
    )
    
    if uploaded_file is not None:
        st.markdown("## Processing Pipeline")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Step 1: Original Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Shadow Document", use_column_width=True)
            
            st.markdown(f"""
            <div class='info-box'>
            <strong>Image Information:</strong><br>
            • Original Size: {image.size[0]} × {image.size[1]} pixels<br>
            • Format: {image.format}<br>
            • Mode: {image.mode}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Processing Controls")
            st.markdown("""
            <div class='info-box'>
            <strong>Processing Steps:</strong><br>
            1. Image preprocessing (resize & normalize)<br>
            2. Shadow detection using U-Net<br>
            3. Shadow mask generation<br>
            4. Shadow removal & correction<br>
            5. Post-processing & result generation
            </div>
            """, unsafe_allow_html=True)
            
            process_button = st.button("🚀 Start Shadow Removal Process", use_container_width=True)
        
        if process_button:
            with st.spinner('🔄 Processing image... Please wait...'):
                # Preprocess image
                img_normalized, original_img = preprocess_image(image)
                input_batch = np.expand_dims(img_normalized, axis=0)
                
                # Predict
                pred_mask, pred_clean_model = model.predict(input_batch, verbose=0)
                pred_mask = pred_mask[0]
                pred_clean_model = pred_clean_model[0]
                
                # Draw boundary
                img_with_boundary = draw_red_boundary(img_normalized, pred_mask)
                
                # Resize to original
                boundary_original = resize_to_original(img_with_boundary / 255.0, original_img.shape)
                clean_original = resize_to_original(pred_clean_model, original_img.shape)
                mask_original = cv2.resize((pred_mask.squeeze() * 255).astype(np.uint8), 
                                          (original_img.shape[1], original_img.shape[0]))
                
                # Calculate metrics
                shadow_percentage = (np.sum(pred_mask > 0.5) / pred_mask.size) * 100
                
                st.success("Processing Complete!")
                
                st.markdown("---")
                st.markdown("## Processing Results & Metrics")
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.markdown("""
                    <div class='metric-card'>
                        <h3 style='color: #0066cc; margin: 0; font-size: 14px;'>Shadow Coverage</h3>
                        <h2 style='color: #333; margin: 10px 0;'>{:.1f}%</h2>
                        <p style='font-size: 11px; color: #666;'>of total area</p>
                    </div>
                    """.format(shadow_percentage), unsafe_allow_html=True)
                
                with metric_col2:
                    st.markdown("""
                    <div class='metric-card'>
                        <h3 style='color: #28a745; margin: 0; font-size: 14px;'>Processing Status</h3>
                        <h2 style='color: #28a745; margin: 10px 0;'>✓</h2>
                        <p style='font-size: 11px; color: #666;'>Complete</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col3:
                    st.markdown("""
                    <div class='metric-card'>
                        <h3 style='color: #0066cc; margin: 0; font-size: 14px;'>Resolution</h3>
                        <h2 style='color: #333; margin: 10px 0; font-size: 18px;'>{}×{}</h2>
                        <p style='font-size: 11px; color: #666;'>pixels</p>
                    </div>
                    """.format(original_img.shape[1], original_img.shape[0]), unsafe_allow_html=True)
                
                with metric_col4:
                    st.markdown("""
                    <div class='metric-card'>
                        <h3 style='color: #ffc107; margin: 0; font-size: 14px;'>Model Used</h3>
                        <h2 style='color: #333; margin: 10px 0; font-size: 18px;'>U-Net</h2>
                        <p style='font-size: 11px; color: #666;'>Deep Learning</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                st.markdown("## Visualization Results")
                
                # Show results in tabs for better organization
                tab1, tab2, tab3 = st.tabs(["🔴 Shadow Detection", "⚫ Shadow Mask", "✨ Final Output"])
                
                with tab1:
                    st.markdown("### Step 2: Shadow Detection (Red Boundary)")
                    st.image(boundary_original, caption="Detected Shadow Regions Marked in Red", use_column_width=True)
                    st.markdown("""
                    <div class='info-box'>
                    The red boundary marks the detected shadow regions identified by the U-Net model. 
                    This visualization helps understand which areas the model considers as shadows.
                    </div>
                    """, unsafe_allow_html=True)
                
                with tab2:
                    st.markdown("### Step 3: Shadow Mask (Binary Segmentation)")
                    st.image(mask_original, caption="Binary Shadow Mask (White = Shadow, Black = No Shadow)", 
                           use_column_width=True)
                    st.markdown("""
                    <div class='info-box'>
                    The shadow mask is a binary image where white pixels indicate shadow regions and black pixels 
                    indicate non-shadow regions. This mask is used for selective shadow correction.
                    </div>
                    """, unsafe_allow_html=True)
                
                with tab3:
                    st.markdown("### Step 4: Model Output (Shadow-Free Result)")
                    st.image(clean_original, caption="U-Net Model Output - Shadow Successfully Removed", 
                           use_column_width=True)
                    st.markdown("""
                    <div class='success-box'>
                    <strong>Final Result:</strong> The shadow has been successfully removed while preserving 
                    document details and text clarity. The output image is ready for OCR or archival use.
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Side-by-side comparison
                st.markdown("## Before & After Comparison")
                
                comp_col1, comp_col2 = st.columns(2)
                
                with comp_col1:
                    st.markdown("### Before (With Shadow)")
                    st.image(original_img, caption="Original Image with Shadow", use_column_width=True)
                
                with comp_col2:
                    st.markdown("### After (Shadow Removed)")
                    st.image(clean_original, caption="Processed Image - Shadow Removed", use_column_width=True)
                
                st.markdown("---")
                
                # Download Section
                st.markdown("## Download Results")
                
                st.markdown("""
                <div class='info-box'>
                Download the processed images for your use. All images are in PNG format for maximum quality.
                </div>
                """, unsafe_allow_html=True)
                
                download_col1, download_col2, download_col3 = st.columns(3)
                
                with download_col1:
                    st.download_button(
                        label="📥 Download Shadow Detection",
                        data=convert_to_downloadable(boundary_original),
                        file_name="shadow_detection_result.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                with download_col2:
                    st.download_button(
                        label="📥 Download Shadow Mask",
                        data=convert_to_downloadable(mask_original),
                        file_name="shadow_mask_binary.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                with download_col3:
                    st.download_button(
                        label="📥 Download Final Output",
                        data=convert_to_downloadable(clean_original),
                        file_name="shadow_free_output.png",
                        mime="image/png",
                        use_container_width=True
                    )
    
    else:
        # No image uploaded - Show instructions
        st.markdown("##How to Use")
        
        st.markdown("""
        <div class='info-box'>
        <strong>Getting Started:</strong> Upload a document image with shadows using the file uploader above 
        to begin the shadow detection and removal process.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Processing Workflow")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <div style='font-size: 32px; color: #0066cc;'>📤</div>
                <h3>1. Upload</h3>
                <p style='font-size: 13px;'>Select shadow document</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <div style='font-size: 32px; color: #28a745;'>🔍</div>
                <h3>2. Detect</h3>
                <p style='font-size: 13px;'>AI identifies shadows</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='metric-card'>
                <div style='font-size: 32px; color: #ffc107;'>⚙️</div>
                <h3>3. Process</h3>
                <p style='font-size: 13px;'>Model removes shadows</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class='metric-card'>
                <div style='font-size: 32px; color: #dc3545;'>💾</div>
                <h3>4. Download</h3>
                <p style='font-size: 13px;'>Get clean result</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📝 Supported File Formats")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='section-card'>
            <h3>Input Formats</h3>
            <ul style='font-size: 15px; line-height: 1.8;'>
                <li># JPG / JPEG</li>
                <li># PNG</li>
                <li># BMP</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='section-card'>
            <h3>Output Formats</h3>
            <ul style='font-size: 15px; line-height: 1.8;'>
                <li># PNG (Lossless)</li>
                <li># Original Resolution</li>
                <li># High Quality</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        

        st.markdown("---")

