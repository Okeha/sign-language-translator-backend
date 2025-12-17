import cv2
import streamlit as st
# Custom CSS for better UI
# st.markdown("""
#     <style>
#     .main {
#         padding: 2rem;
#     }
#     .stButton>button {
#         width: 100%;
#         background-color: #4CAF50;
#         color: white;rt numpy as np
import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import time
from collections import deque
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import threading
import numpy as np
import cv2

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Sign Language Detector",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# # Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.75rem;
        font-size: 1.1rem;
        border-radius: 10px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .top5-item {
        background-color: #ffffff;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 3px solid #4CAF50;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    h1 {
        color: #667eea;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# WebRTC Configuration for camera access
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    }
)

class VideoFrameCollector:
    """Collects video frames from WebRTC stream for sign language prediction"""
    
    def __init__(self, max_frames=60):
        self.frames = []
        self.max_frames = max_frames
        self.is_recording = False
        self.lock = threading.Lock()
    
    def start_recording(self):
        with self.lock:
            self.frames = []
            self.is_recording = True
    
    def stop_recording(self):
        with self.lock:
            self.is_recording = False
    
    def add_frame(self, frame):
        with self.lock:
            if self.is_recording and len(self.frames) < self.max_frames:
                self.frames.append(frame)
                return True
            return False
    
    def get_frames(self):
        with self.lock:
            return self.frames.copy()
    
    def get_frame_count(self):
        with self.lock:
            return len(self.frames)
    
    def clear_frames(self):
        with self.lock:
            self.frames = []
            self.is_recording = False

class VideoProcessor:
    """Process video frames from WebRTC stream"""
    
    def __init__(self):
        self.frame_collector = None
    
    def set_collector(self, collector):
        self.frame_collector = collector
    
    def recv(self, frame):
        img = frame.to_ndarray(format="rgb24")
        
        # Add frame to collector if recording
        if self.frame_collector and self.frame_collector.is_recording:
            self.frame_collector.add_frame(img)
        
        # Return the frame to display
        return av.VideoFrame.from_ndarray(img, format="rgb24")

@st.cache_resource
def load_model(model_path):
    """Load the fine-tuned VideoMAE model"""
    try:
        model = VideoMAEForVideoClassification.from_pretrained(model_path)
        processor = VideoMAEImageProcessor.from_pretrained(model_path)
        model.eval()
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        return model, processor, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def process_video_frames(frames, processor, num_frames=16):
    """Process video frames for model input"""
    # Sample frames uniformly
    total_frames = len(frames)
    if total_frames < num_frames:
        # Duplicate frames if not enough
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    
    sampled_frames = [frames[i] for i in indices]
    
    # Process frames
    inputs = processor(sampled_frames, return_tensors="pt")
    return inputs

def predict_sign(frames, model, processor, device):
    """Make prediction from video frames"""
    start_time = time.time()
    
    # Process frames
    inputs = process_video_frames(frames, processor)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get top 5 predictions
        top5_probs, top5_indices = torch.topk(probs, 5, dim=-1)
        top5_probs = top5_probs[0].cpu().numpy()
        top5_indices = top5_indices[0].cpu().numpy()
        
        # Get predicted class
        predicted_idx = torch.argmax(logits, dim=-1).item()
    
    latency = time.time() - start_time
    
    # Get class names
    id2label = model.config.id2label
    predicted_gloss = id2label[predicted_idx]
    top5_glosses = [(id2label[idx], prob) for idx, prob in zip(top5_indices, top5_probs)]
    
    return predicted_gloss, top5_glosses, latency

def capture_video_from_webcam(duration=3, fps=30):
    """Capture video from webcam"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        return None
    
    frames = []
    start_time = time.time()
    
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    return frames

def main():
    # Header
    st.markdown("<h1>🤟 Sign Language Detector</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem;'>Real-time sign language recognition using VideoMAE</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model path
        model_path = st.text_input(
            "Model Path",
            value="../model/finetune/videomae/video_mae_finetuned_final",
            help="Path to the fine-tuned VideoMAE model"
        )
        
        # Recording duration
        duration = st.slider(
            "Recording Duration (seconds)",
            min_value=1,
            max_value=5,
            value=2,
            help="Duration to record sign language gesture"
        )
        
        st.divider()
        
        # Device info
        device_type = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        st.info(f"**Device:** {device_type}")
        
        st.divider()
        
        # Instructions
        st.markdown("""
        ### 📝 Instructions
        
        **Live Camera Mode:**
        1. Grant camera permissions when browser asks
        2. Click **Start Recording**
        3. Perform your sign language gesture
        4. Click **Stop Recording** (or wait for auto-stop)
        5. Click **Make Prediction** to see results
        
        **Upload Video Mode:**
        - Upload pre-recorded .mp4/.avi files
        - No camera needed!
        
        **Note:** Camera access works via WebRTC (browser-based), so it works everywhere - even in WSL2, Docker, or cloud servers! 🎉
        """)
    
    # Load model
    model, processor, device = load_model(model_path)
    
    if model is None:
        st.error("❌ Failed to load model. Please check the model path.")
        return
    
    st.success(f"✅ Model loaded successfully on {device}")
    
    st.info("""
    ### 🎥 Camera Access via Browser
    
    This app uses **WebRTC** for camera access - works on WSL2, Docker, and remote servers!
    
    - Click "Start Recording" to capture frames
    - Click "Stop Recording" when done
    - No `/dev/video` devices needed - camera access happens in your browser!
    """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📹 Input Source")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["📷 Live Camera", "📁 Upload Video"],
            horizontal=True
        )
        
        # Placeholder for video display
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        
        if input_method == "📷 Live Camera":
            # Live camera recording using WebRTC
            st.markdown("---")
            
            st.info("📹 **Camera Preview** - Grant camera permissions when prompted")
            
            # Initialize frame collector in session state FIRST
            if 'frame_collector' not in st.session_state:
                st.session_state.frame_collector = VideoFrameCollector(max_frames=int(duration * 30))
            
            # Update max frames if duration changes
            st.session_state.frame_collector.max_frames = int(duration * 30)
            
            # Create a class that captures the frame_collector at initialization
            class VideoProcessorFactory:
                def __init__(self, collector):
                    self.collector = collector
                
                def __call__(self):
                    processor = VideoProcessor()
                    processor.set_collector(self.collector)
                    return processor
            
            # WebRTC streamer for camera access
            try:
                webrtc_ctx = webrtc_streamer(
                    key="sign-language-camera",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTC_CONFIGURATION,
                    video_processor_factory=VideoProcessorFactory(st.session_state.frame_collector),
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )
            except Exception as e:
                st.error(f"❌ WebRTC Error: {e}")
                st.info("Try refreshing the page or using Upload Video mode instead.")
                webrtc_ctx = None
            
            st.markdown("---")
            
            # Recording controls
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                if st.button("🎥 Start Recording", key="start_record", use_container_width=True):
                    st.session_state.frame_collector.start_recording()
                    st.session_state.recording_start_time = time.time()
                    st.rerun()
            
            with col_b:
                if st.button("⏹️ Stop Recording", key="stop_record", use_container_width=True):
                    st.session_state.frame_collector.stop_recording()
                    frames = st.session_state.frame_collector.get_frames()
                    if len(frames) > 0:
                        st.session_state.frames = frames
                        st.session_state.has_recording = True
                        st.success(f"✅ Captured {len(frames)} frames!")
                    else:
                        st.warning("⚠️ No frames captured")
                    st.rerun()
            
            with col_c:
                if st.button("🗑️ Clear", key="clear_frames", use_container_width=True):
                    st.session_state.frame_collector.clear_frames()
                    if 'frames' in st.session_state:
                        del st.session_state.frames
                    if 'has_recording' in st.session_state:
                        del st.session_state.has_recording
                    # Clear live predictions
                    if 'live_prediction' in st.session_state:
                        del st.session_state.live_prediction
                    if 'live_confidence' in st.session_state:
                        del st.session_state.live_confidence
                    if 'live_top5' in st.session_state:
                        del st.session_state.live_top5
                    if 'live_latency' in st.session_state:
                        del st.session_state.live_latency
                    st.rerun()
            
            # Show recording status
            if st.session_state.frame_collector.is_recording:
                current_frames = st.session_state.frame_collector.get_frame_count()
                max_frames = st.session_state.frame_collector.max_frames
                progress = current_frames / max_frames if max_frames > 0 else 0
                
                st.progress(min(progress, 1.0))
                st.markdown(f"""
                    <div style='text-align: center; color: #ff4444; font-weight: bold; font-size: 1.2rem;'>
                        🔴 RECORDING... {current_frames}/{max_frames} frames
                    </div>
                """, unsafe_allow_html=True)
                
                # Continuous inference every ~2 seconds (60 frames at 30fps)
                inference_interval = 60  # frames
                
                # Run inference every time we have a new batch of 60 frames
                if current_frames >= inference_interval:
                    # Check if we should run inference (every 60 frames = every 2 seconds)
                    frames_since_last_inference = current_frames % inference_interval
                    
                    # Run inference at each 60-frame interval
                    if frames_since_last_inference < 5:  # Small window to catch the interval
                        all_frames = st.session_state.frame_collector.get_frames()
                        recent_frames = all_frames[-inference_interval:]
                        
                        # Run inference on recent frames
                        try:
                            predicted_gloss, top5_glosses, latency = predict_sign(
                                recent_frames,
                                model,
                                processor,
                                device
                            )
                            
                            # Store live prediction
                            st.session_state.live_prediction = predicted_gloss
                            st.session_state.live_confidence = top5_glosses[0][1] * 100
                            st.session_state.live_top5 = top5_glosses
                            st.session_state.live_latency = latency
                        except Exception as e:
                            pass  # Silent fail during recording
                
                # Auto-stop when max frames reached
                if current_frames >= max_frames:
                    st.session_state.frame_collector.stop_recording()
                    st.session_state.frames = st.session_state.frame_collector.get_frames()
                    st.session_state.has_recording = True
                    st.success(f"✅ Recording complete! Captured {current_frames} frames")
                    
                    # Keep live prediction visible after recording stops
                    # Don't clear it - let user see the final prediction
                    
                    time.sleep(0.5)
                    st.rerun()
                else:
                    # Auto-refresh to update counter
                    time.sleep(0.1)
                    st.rerun()
            
            # Show captured frames info
            if hasattr(st.session_state, 'frames') and len(st.session_state.frames) > 0:
                status_placeholder.info(f"📹 {len(st.session_state.frames)} frames ready for prediction")
                
                # Show last captured frame as thumbnail in expander - MOVED TO TOP
                with st.expander("👁️ View Individual Frames", expanded=False):
                    # Frame selector
                    frame_idx = st.slider(
                        "Select frame to view",
                        0,
                        len(st.session_state.frames) - 1,
                        len(st.session_state.frames) - 1,
                        key="frame_selector"
                    )
                    st.image(
                        st.session_state.frames[frame_idx],
                        caption=f"Frame {frame_idx + 1}/{len(st.session_state.frames)}",
                        channels="RGB",
                        width="stretch"
                    )
                
                # Show video playback
                st.markdown("#### 🎬 Recorded Video Playback")
                
                # Create video from frames - Use content hash to detect new recordings
                # Create hash from first, middle, and last frame to detect content changes
                current_frame_hash = hash((
                    st.session_state.frames[0].tobytes(),
                    st.session_state.frames[len(st.session_state.frames)//2].tobytes(),
                    st.session_state.frames[-1].tobytes(),
                    len(st.session_state.frames)
                ))
                need_new_video = (
                    'recorded_video_path' not in st.session_state or 
                    st.session_state.get('last_frame_hash') != current_frame_hash or
                    not os.path.exists(st.session_state.get('recorded_video_path', ''))
                )
                
                if need_new_video:
                    try:
                        # Clean up old video FIRST
                        if 'recorded_video_path' in st.session_state and os.path.exists(st.session_state.recorded_video_path):
                            try:
                                os.unlink(st.session_state.recorded_video_path)
                            except:
                                pass
                        
                        # Save frames as video
                        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', mode='wb')
                        temp_video_path = temp_video.name
                        temp_video.close()
                        
                        fps = len(st.session_state.frames) / duration if duration > 0 else 30
                        fps = max(10, min(fps, 60))  # Ensure reasonable FPS
                        
                        success = False
                        
                        # Try imageio first (more reliable for web)
                        if IMAGEIO_AVAILABLE:
                            try:
                                with imageio.get_writer(temp_video_path, fps=fps, codec='libx264', pixelformat='yuv420p') as writer:
                                    for frame in st.session_state.frames:
                                        writer.append_data(frame)
                                success = True
                            except Exception as e:
                                st.warning(f"⚠️ imageio failed: {e}, trying cv2...")
                        
                        # Fallback to cv2
                        if not success:
                            height, width = st.session_state.frames[0].shape[:2]
                            
                            # Try H264 codec first
                            fourcc = cv2.VideoWriter_fourcc(*'avc1')
                            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
                            
                            if not out.isOpened():
                                # Fallback to mp4v
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
                            
                            if out.isOpened():
                                for frame in st.session_state.frames:
                                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                    out.write(frame_bgr)
                                out.release()
                                success = True
                        
                        # Verify video file
                        if success and os.path.exists(temp_video_path) and os.path.getsize(temp_video_path) > 0:
                            st.session_state.recorded_video_path = temp_video_path
                            st.session_state.last_frame_hash = current_frame_hash
                        else:
                            st.error("❌ Failed to create video file")
                            st.info("💡 Video playback unavailable, but predictions still work!")
                    
                    except Exception as e:
                        st.error(f"❌ Error creating video: {e}")
                        st.info("💡 Try installing imageio: `pip install imageio imageio-ffmpeg`")
                
                # Display video player
                if 'recorded_video_path' in st.session_state and os.path.exists(st.session_state.recorded_video_path):
                    try:
                        with open(st.session_state.recorded_video_path, 'rb') as video_file:
                            video_bytes = video_file.read()
                            if len(video_bytes) > 0:
                                st.video(video_bytes)
                            else:
                                st.warning("⚠️ Video file is empty")
                    except Exception as e:
                        st.error(f"❌ Error loading video: {e}")
        
        else:
            # Video upload
            st.markdown("---")
            uploaded_file = st.file_uploader(
                "Upload a video file",
                type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
                help="Upload a short video (1-5 seconds) of sign language gesture"
            )
            
            if uploaded_file is not None:
                # Save uploaded file temporarily (keep it for playback)
                if 'video_path' not in st.session_state or st.session_state.get('last_uploaded_name') != uploaded_file.name:
                    # Clean up old temp file if exists
                    if 'video_path' in st.session_state and os.path.exists(st.session_state.video_path):
                        try:
                            os.unlink(st.session_state.video_path)
                        except:
                            pass
                    
                    # Save new file
                    suffix = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    
                    st.session_state.video_path = tmp_path
                    st.session_state.last_uploaded_name = uploaded_file.name
                    
                    # Read video frames
                    cap = cv2.VideoCapture(tmp_path)
                    frames = []
                    
                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        total_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(frame_rgb)
                        
                        cap.release()
                        
                        if len(frames) > 0:
                            duration_sec = len(frames) / fps if fps > 0 else len(frames) / 30
                            st.success(f"✅ Loaded {len(frames)} frames from video ({duration_sec:.1f}s @ {fps:.1f} fps)")
                            
                            # Store frames
                            st.session_state.frames = frames
                            st.session_state.has_recording = True
                        else:
                            st.error("❌ Could not read frames from video")
                    else:
                        st.error("❌ Could not open video file")
                
                # Display video player
                if 'video_path' in st.session_state and os.path.exists(st.session_state.video_path):
                    st.markdown("#### 🎬 Uploaded Video")
                    with open(st.session_state.video_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                    
                    status_placeholder.info(f"📹 {len(st.session_state.frames)} frames loaded and ready for prediction")
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        # Show live prediction during AND after recording
        if hasattr(st.session_state, 'live_prediction'):
            # Determine if we're currently recording
            is_recording = hasattr(st.session_state, 'frame_collector') and st.session_state.frame_collector.is_recording
            
            # Show live prediction with dynamic styling
            if is_recording:
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%); 
                                padding: 1.5rem; border-radius: 10px; color: white; text-align: center; margin: 1rem 0;'>
                        <h3 style='margin: 0; font-size: 1.8rem;'>🔴 LIVE: "{st.session_state.live_prediction}"</h3>
                        <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Confidence: {st.session_state.live_confidence:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 1.5rem; border-radius: 10px; color: white; text-align: center; margin: 1rem 0;'>
                        <h3 style='margin: 0; font-size: 1.8rem;'>"{st.session_state.live_prediction}"</h3>
                        <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Confidence: {st.session_state.live_confidence:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Show top 5 if available
            if hasattr(st.session_state, 'live_top5'):
                st.markdown("### 📊 Live Top 5 Predictions")
                for i, (gloss, prob) in enumerate(st.session_state.live_top5):
                    confidence_pct = prob * 100
                    st.markdown(f"""
                        <div class='top5-item'>
                            <div>
                                <strong style='color: black;'>#{i+1} {gloss}</strong>
                            </div>
                            <div style='text-align: right;'>
                                <span style='background-color: #4CAF50; color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.9rem;'>
                                    {confidence_pct:.1f}%
                                </span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(float(prob))
            
            # Show latency if available
            if hasattr(st.session_state, 'live_latency'):
                st.markdown(f"""
                    <div class='metric-card'>
                        <h4 style='margin: 0; color: #667eea;'>⚡ Inference Time</h4>
                        <p style='font-size: 1.8rem; color: black; font-weight: bold; margin: 0.5rem 0;'>{st.session_state.live_latency*1000:.0f} ms</p>
                        <p style='margin: 0; color: black; font-size: 0.9rem;'>
                            {1/st.session_state.live_latency:.1f} predictions/second
                        </p>
                    </div>
                """, unsafe_allow_html=True)
        
        elif hasattr(st.session_state, 'frame_collector') and st.session_state.frame_collector.is_recording:
            st.info("🎥 Recording... Inference will start after 2 seconds (60 frames)")
        
        # Predict button (for final prediction after recording stops) - OPTIONAL now
        if not (hasattr(st.session_state, 'frame_collector') and st.session_state.frame_collector.is_recording):
            if st.button("🔮 Re-run Full Prediction", key="predict_btn", disabled=not hasattr(st.session_state, 'has_recording')):
                if hasattr(st.session_state, 'frames'):
                    with st.spinner("Analyzing sign language gesture..."):
                        predicted_gloss, top5_glosses, latency = predict_sign(
                            st.session_state.frames,
                            model,
                            processor,
                            device
                        )
                        
                        # Store results
                        st.session_state.prediction = predicted_gloss
                        st.session_state.top5 = top5_glosses
                        st.session_state.latency = latency
                        
                        # Also update live predictions
                        st.session_state.live_prediction = predicted_gloss
                        st.session_state.live_confidence = top5_glosses[0][1] * 100
                        st.session_state.live_top5 = top5_glosses
                        st.session_state.live_latency = latency
        
        # No separate "Display results" section - everything is shown via live predictions above
        else:
            st.info("👆 Record a video and click 'Make Prediction' to see results")
    
    # Footer
    st.divider()
    st.markdown("""
        <p style='text-align: center; color: #999; font-size: 0.9rem;'>
            Built with Streamlit • Powered by VideoMAE • 2025
        </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
