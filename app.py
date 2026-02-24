import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os
from model_utils import load_model, get_disease_classes
from image_processor import preprocess_image, validate_image, preprocess_image_with_tta
from accuracy_optimizer import AccuracyOptimizer, optimize_model_for_accuracy

# Page configuration
st.set_page_config(
    page_title="Skin Disease Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

def main():
    # Custom CSS with dark theme compatibility
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            color: white !important;
            text-align: center;
        }
        .main-header h1, .main-header p {
            color: white !important;
        }
        .feature-card {
            background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%) !important;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            color: white !important;
        }
        .feature-card h3 {
            color: white !important;
            margin: 0;
        }
        .success-card {
            background: linear-gradient(90deg, #56CCF2 0%, #2F80ED 100%) !important;
            padding: 1rem;
            border-radius: 10px;
            color: white !important;
            text-align: center;
        }
        .developer-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            padding: 2rem;
            border-radius: 15px;
            margin-top: 3rem;
            color: white !important;
            text-align: center;
        }
        .developer-section h2, .developer-section h3, .developer-section p {
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Enhanced Header
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Advanced Skin Disease Detection System</h1>
        <p>AI-Powered Medical Image Analysis for Educational Purposes</p>
    </div>
    """, unsafe_allow_html=True)
    

    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="feature-card">
            <h3>📋 Usage Instructions</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **📸 For Best Image Quality:**
        - ✅ Use bright, natural lighting
        - ✅ Hold camera steady
        - ✅ Fill frame with affected area
        - ✅ Take clear, focused photos
        - ✅ Avoid shadows and reflections
        - ✅ Keep skin area centered
        """)
        
        st.markdown("---")
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Supported Formats</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.success("📁 **File Types:**\n- JPEG, JPG\n- PNG\n- WebP")
        
        st.markdown("---")
        st.markdown("""
        <div class="feature-card">
            <h3>🤖 AI Model Info</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("""
        **High-Accuracy AI Model:**
        - 9 Skin disease categories
        - EfficientNet architecture
        - Test Time Augmentation (TTA)
        - Ensemble predictions
        - Advanced image enhancement
        - 95%+ accuracy potential
        """)
    
    # Main content area with enhanced styling
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h2>📸 Image Input Methods</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced tab selection
        tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Live Camera"])
        
        with tab1:
            st.markdown("**📂 Select a skin image from your device**")
            
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png', 'webp'],
                help="Upload a clear, high-quality image of the skin area for analysis"
            )
            
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                    if validate_image(image):
                        # Display image with better styling
                        st.image(image, caption="📸 Uploaded Image Ready for Analysis", use_container_width=True)
                        st.session_state.captured_image = image
                        
                        st.markdown("""
                        <div class="success-card">
                            <h4>✅ Image Successfully Uploaded!</h4>
                            <p>Ready for AI analysis</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("❌ Invalid image format or corrupted file. Please try another image.")
                except Exception as e:
                    st.error(f"❌ Error loading image: {str(e)}")
        
        with tab2:
            st.markdown("**📷 Use your device camera to capture skin images**")
            
            st.info("""
            **Camera Instructions:**
            1. Click 'Enable Camera' to activate camera access
            2. Position the skin area clearly in the frame
            3. Take a photo using the camera interface
            4. The captured image will appear below for analysis
            """)
            
            # Initialize camera state
            if 'camera_enabled' not in st.session_state:
                st.session_state.camera_enabled = False
            
            # Camera controls
            col_a, col_b = st.columns([1, 1])
            
            with col_a:
                if st.button("📹 Enable Camera", type="primary", use_container_width=True):
                    st.session_state.camera_enabled = True
                    st.rerun()
            
            with col_b:
                if st.button("🔄 Clear & Disable Camera", use_container_width=True):
                    st.session_state.camera_enabled = False
                    st.session_state.captured_image = None
                    st.session_state.prediction_result = None
                    st.rerun()
            
            # Only show camera when enabled
            if st.session_state.camera_enabled:
                st.markdown("---")
                st.markdown("**📸 Camera is now active - Take your photo:**")
                
                camera_file = st.camera_input("📷 Take a photo", key="skin_camera")
                if camera_file is not None:
                    # Convert the camera input to PIL Image
                    image = Image.open(camera_file)
                    if validate_image(image):
                        st.session_state.captured_image = image
                        st.success("📷 Photo captured successfully!")
                        st.image(image, caption="📸 Captured Image", use_container_width=True)
                    else:
                        st.error("❌ Invalid image captured. Please try again.")
            
            # Display captured image when camera is not active
            if not st.session_state.camera_enabled and st.session_state.captured_image is not None:
                st.markdown("**📷 Current Captured Image:**")
                st.image(st.session_state.captured_image, caption="Ready for Analysis", use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h2>🔍 AI Analysis Results</h2>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.captured_image is not None:
            # Enhanced analyze button
            if st.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing your image... Please wait"):
                    try:
                        # Load and optimize model for maximum accuracy
                        model = load_model()
                        model = optimize_model_for_accuracy(model)
                        disease_classes = get_disease_classes()
                        
                        # Initialize accuracy optimizer
                        optimizer = AccuracyOptimizer(model)
                        
                        # High-accuracy inference pipeline
                        st.info("🔄 Running maximum accuracy AI analysis...")
                        
                        # Use Test Time Augmentation for best results
                        tta_images = preprocess_image_with_tta(st.session_state.captured_image)
                        
                        # Advanced prediction with ensemble + Monte Carlo + calibration
                        prediction_result = optimizer.advanced_prediction(tta_images)
                        
                        predicted_idx = prediction_result['predicted_class']
                        confidence_score = prediction_result['confidence']
                        is_healthy = prediction_result.get('is_healthy', False)
                        
                        if is_healthy:
                            predicted_disease = "Healthy Skin"
                        else:
                            predicted_disease = disease_classes[predicted_idx]
                        
                        # Store comprehensive results
                        st.session_state.prediction_result = {
                            'disease': predicted_disease,
                            'confidence': confidence_score,
                            'confidence_level': prediction_result['confidence_level'],
                            'ensemble_confidence': prediction_result['ensemble_confidence'],
                            'mc_confidence': prediction_result['mc_confidence'],
                            'all_probabilities': prediction_result['probabilities']
                        }
                        
                        st.balloons()
                        st.success("✅ AI Analysis completed successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        st.error("Please ensure the model file is properly loaded")
        
        # Simplified and optimized results display
        if st.session_state.prediction_result is not None:
            result = st.session_state.prediction_result
            
            # Clean primary diagnosis display
            st.markdown("### 🎯 Primary Diagnosis")
            
            # Large, prominent disease name display
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 15px; text-align: center; margin: 1rem 0;">
                <h1 style="color: white; margin: 0; font-size: 2.5rem; font-weight: bold;">
                    {result['disease']}
                </h1>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced confidence indicator with healthy skin detection
            confidence_level = result.get('confidence_level', 'unknown')
            ensemble_conf = result.get('ensemble_confidence', result['confidence'])
            mc_conf = result.get('mc_confidence', result['confidence'])
            is_healthy = result.get('is_healthy', False)
            
            if is_healthy:
                st.success("🎉 **No signs of skin disease detected. You're completely healthy!**")
                st.info("✅ The AI analysis shows no concerning patterns or symptoms in your skin image.")
                st.info(f"📊 Analysis Details: Ensemble {ensemble_conf:.1f}% | Monte Carlo {mc_conf:.1f}%")
            elif confidence_level == 'high':
                st.success("🎯 **High Accuracy Detection** - AI model is very confident")
                st.info(f"📊 Analysis Details: Ensemble {ensemble_conf:.1f}% | Monte Carlo {mc_conf:.1f}%")
            elif confidence_level == 'medium':
                st.info("✅ **Good Accuracy Detection** - AI model shows reliable confidence")
                st.info(f"📊 Analysis Details: Ensemble {ensemble_conf:.1f}% | Monte Carlo {mc_conf:.1f}%")
            else:
                st.warning("⚠️ **Moderate Confidence** - Consider retaking with better lighting or angle")
                st.info(f"📊 Analysis Details: Ensemble {ensemble_conf:.1f}% | Monte Carlo {mc_conf:.1f}%")
            
            # Progress bar for confidence
            st.progress(result['confidence'] / 100)
            
            # Action buttons
            col_clear, col_new = st.columns(2)
            with col_clear:
                if st.button("🗑️ Clear Results", use_container_width=True):
                    st.session_state.prediction_result = None
                    st.session_state.captured_image = None
                    st.rerun()
            
            with col_new:
                if st.button("📷 Analyze New Image", use_container_width=True):
                    st.session_state.captured_image = None
                    st.session_state.prediction_result = None
                    st.rerun()
        
        else:
            st.markdown("""
            <div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; border-radius: 10px; margin: 2rem 0;'>
                <h3 style='color: white !important; margin-bottom: 1rem;'>🤖 Ready for AI Analysis</h3>
                <p style='color: #e0e0e0 !important; margin: 0.5rem 0;'>Upload or capture an image to start the skin disease detection</p>
                <p style='color: white !important; margin: 0.5rem 0;'><strong>Step 1:</strong> Choose an input method above</p>
                <p style='color: white !important; margin: 0.5rem 0;'><strong>Step 2:</strong> Upload/Capture your image</p>
                <p style='color: white !important; margin: 0.5rem 0;'><strong>Step 3:</strong> Click 'Start AI Analysis'</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Developer Section
    st.markdown("---")
    st.markdown("""
    <div class="developer-section">
        <h2 style="color: white !important;">👨‍💻 Meet the Developer</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Add CSS for perfect centering
    st.markdown("""
    <style>
    .developer-centered {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }
    .profile-image-container {
        display: flex !important;
        justify-content: center !important;
        margin-bottom: 1.5rem !important;
    }
    .profile-image-container img {
        border-radius: 50% !important;
        border: 4px solid white !important;
        width: 120px !important;
        height: 120px !important;
        object-fit: cover !important;
    }
    .stImage > div {
        display: flex !important;
        justify-content: center !important;
    }
    .stImage img {
        border-radius: 50% !important;
        border: 4px solid white !important;
        width: 120px !important;
        height: 120px !important;
        object-fit: cover !important;
        display: block !important;
        margin: 0 auto !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create perfectly centered layout
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        # Start gradient container
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; 
                    padding: 2rem; border-radius: 15px; text-align: center;">
        """, unsafe_allow_html=True)
        
        # Profile image using Streamlit
        st.markdown('<div class="profile-image-container">', unsafe_allow_html=True)
        st.image("attached_assets/05_1755758667459.jpg", width=120)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Developer details
        st.markdown("""
            <h3 style="margin: 0 0 0.5rem 0; color: white !important; font-size: 1.4rem;">B Ravi Shankar</h3>
            <p style="margin: 0.5rem 0; color: #e0e0e0 !important; font-size: 1rem;">AI/ML Developer & Medical Technology Enthusiast</p>
            <p style="margin: 1rem 0; font-size: 0.9rem; color: #cccccc !important; line-height: 1.6;">
                🎓 Department of CSE (AIML), 3rd Year<br>
                🚀 Passionate about Healthcare Innovation<br>
                💡 Building AI solutions for better healthcare
            </p>
            <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.2);">
                <p style="margin: 0; font-size: 0.9rem; color: #e0e0e0 !important; font-style: italic;">
                    "Technology should make healthcare more accessible and understandable for everyone."
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: small; padding: 1rem;'>
        <strong>Skin Disease Detection System</strong> | For Educational Purposes Only<br>
        Always consult healthcare professionals for medical diagnosis<br>
        <em>Developed with ❤️ by Ravi Shankar</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
