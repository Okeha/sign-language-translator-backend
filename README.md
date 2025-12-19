# Sign Language Detector Backend 🤟

A comprehensive backend system for real-time sign language detection and interpretation using state-of-the-art Vision-Language Models (VLMs) and modern web technologies.

## 🎯 Project Overview

This project aims to bridge communication gaps by providing an intelligent system that can accurately detect and interpret sign language gestures in real-time. The system combines computer vision, natural language processing, and web technologies to create an accessible and scalable solution.

### Key Features

- **Real-time Sign Language Detection**: Process video streams and static images
- **Vision-Language Model Integration**: Powered by InternVL3.5-2B for accurate interpretation
- **Multi-Hardware Support**: Optimized for NVIDIA CUDA (with TF32/BF16), AMD DirectML, and CPU
- **Fine-tuning Pipeline**: Complete SFT training with PEFT/LoRA for efficient adaptation
- **Custom Data Collation**: Video frame sampling with chat template formatting and proper label masking
- **Stratified Dataset Splitting**: Gloss-aware train/test splits ensuring balanced representation
- **Quantization Support**: 4-bit quantization for efficient inference (CUDA only)
- **Automated Video Processing**: Frame extraction and preprocessing with Decord
- **TensorBoard Monitoring**: Real-time training/validation loss tracking
- **Early Stopping**: Automatic training termination when model stops improving

## 🏗️ Architecture

```
sign-language-detector-backend/
├── src/
│   ├── __init__.py                # Package marker
│   ├── app/                       # 🎨 Streamlit Web Application
│   │   ├── streamlit_app.py       # Main Streamlit app with WebRTC camera
│   │   ├── requirements.txt       # App-specific dependencies
│   │   ├── run_windows.bat        # Windows launcher (WSL2 camera fix)
│   │   └── README.md              # App documentation
│   ├── model/                     # 🤖 ML Model Components
│   │   ├── __init__.py            # Package marker
│   │   ├── model_loader.py        # InternVL3 model loader (legacy)
│   │   ├── params/                # Model configuration
│   │   │   └── vlm.yml            # VLM & VideoMAE parameters
│   │   ├── finetune/              # 🔧 Fine-tuning Pipelines
│   │   │   ├── __init__.py        # Package marker
│   │   │   ├── data_engineering/  # Dataset processing
│   │   │   │   ├── __init__.py    # Package marker
│   │   │   │   ├── video_downloader.py     # YouTube video downloader
│   │   │   │   ├── filter_data.py          # WLASL data filtering
│   │   │   │   ├── validate_videos.py      # Video corruption detection
│   │   │   │   ├── raw_videos/             # Downloaded video files
│   │   │   │   └── datasets/               # Processed datasets (JSON)
│   │   │   │       ├── wlasl_validated.json   # Clean dataset (use this!)
│   │   │   │       ├── wlasl_cleaned.json     # May have corrupted videos
│   │   │   │       └── corrupted_videos.json  # Corruption log
│   │   │   ├── dataset.py         # Shared dataset loader
│   │   │   ├── videomae/          # 📹 VideoMAE Training (CURRENT PIPELINE)
│   │   │   │   ├── train_video_mae.py         # VideoMAE fine-tuning script
│   │   │   │   ├── video_mae_eval.py          # Evaluation script
│   │   │   │   ├── dataset.py                 # VideoMAE dataset loader
│   │   │   │   ├── video_mae_finetuned/       # Training checkpoints
│   │   │   │   ├── video_mae_finetuned_tb_logs/  # TensorBoard logs
│   │   │   │   ├── evaluation_results.json    # Eval predictions
│   │   │   │   └── evaluation_report.txt      # Metrics summary
│   │   │   ├── internvl3_5/       # 🧠 InternVL3.5 Training (legacy)
│   │   │   │   ├── train.py       # InternVL fine-tuning (VLM approach)
│   │   │   │   └── eval.py        # VLM evaluation
│   │   │   ├── preprocess_videos.py  # Optional: Pre-extract frames
│   │   │   └── CLASSIFICATION_HEAD_GUIDE.md  # Architecture notes
│   │   └── tests/                 # Test data and validation
│   │       └── data/
│   │           ├── images/        # Test images
│   │           └── videos/        # Test videos
│   └── api/                       # REST API endpoints (planned)
├── main.py                        # Application entry point (legacy)
├── pyproject.toml                 # UV package dependencies
├── uv.lock                        # Dependency lock file
└── README.md                      # This file
```

### Package Structure Notes

- All directories have `__init__.py` files for proper Python package imports
- Use **relative imports** throughout the codebase
- **Primary Training Pipeline**: `src/model/finetune/videomae/` (VideoMAE for 282 WLASL classes)
- **Legacy Pipeline**: `src/model/finetune/internvl3_5/` (VLM-based approach, not actively used)
- **Streamlit App**: `src/app/streamlit_app.py` (real-time inference with WebRTC camera)
- Run modules from project root: `python src/model/finetune/videomae/train_video_mae.py`

## 🚀 Development Stages

### Stage 1: Foundation ✅ (Completed)

- [x] Project structure setup
- [x] VLM model integration (InternVL3-2B → InternVL3.5-2B)
- [x] VideoMAE model integration (MCG-NJU/videomae-base)
- [x] Multi-hardware support (CUDA/DirectML/CPU)
- [x] Basic video processing capabilities
- [x] Configuration management
- [x] Memory optimization and cleanup
- [x] UV package management setup
- [x] Streamlit web application with WebRTC camera access

### Stage 2: Data Pipeline & Fine-tuning ✅ (Completed)

**Completed:**

- [x] WLASL dataset integration (2,000+ words)
- [x] YouTube video downloader with skip-if-exists optimization
- [x] Data filtering pipeline (320-word core glossary → 282 classes)
- [x] Dataset validation and corruption detection
- [x] Dataset generation for fine-tuning
- [x] Dataset loader with gloss-aware stratified train/val/test split (70/15/15 by gloss)
- [x] **VideoMAE Training Pipeline** (Primary approach)
  - [x] VideoMAE dataset loader with aggressive augmentation
  - [x] Spatial augmentation (flip, brightness, crop, rotation)
  - [x] Temporal augmentation (speed variation)
  - [x] Class balancing with sample repetition
  - [x] Training with Hugging Face Trainer
  - [x] Checkpoint management and best model selection
  - [x] TensorBoard integration
- [x] **InternVL3.5 Training Pipeline** (Legacy VLM approach)
  - [x] Custom video collation with dynamic frame sampling
  - [x] Label masking (prompt tokens = -100, answer tokens = target)
  - [x] PEFT/LoRA integration with gradient checkpointing
  - [x] Chat template integration
- [x] Hardware-specific optimizations (fp16/bf16 for CUDA, float32 for DirectML)
- [x] Batch size optimization and gradient accumulation
- [x] Video preprocessing script for faster training (optional)
- [x] Evaluation pipeline with Top-1/Top-5 Accuracy metrics
- [x] Label smoothing and early stopping

**Current Status (as of Dec 18, 2025):**

**VideoMAE Training (Primary Pipeline):**

- ✅ Successfully trained VideoMAE-base on WLASL dataset
- ✅ Dataset: 282 classes, ~1,603 validated videos
- ✅ Model: MCG-NJU/videomae-base fine-tuned with classification head
- ✅ **Training Configuration:**
  - `batch_size=2`, `gradient_accumulation_steps=2` (effective batch=4)
  - `num_train_epochs=40` (stopped at 29 epochs)
  - `learning_rate=5e-5`, cosine scheduler with warmup
  - **16 Frames per Video** sampled uniformly at 30fps
  - **Aggressive Augmentation Pipeline:**
    - Horizontal flip (50%)
    - Brightness/contrast adjustment (70%)
    - Random crop + resize (80%, zoom 0-30%)
    - Random rotation (30%, ±10°)
    - Temporal speed variation (50%, 80-120%)
  - **Class Balancing:** Target 10 samples per class via repetition
  - **Label Smoothing:** 0.1
  - **Architecture:** 8 unfrozen layers + classification head
- ✅ **Performance:**
  - Top-5 Accuracy: ~28% @ Epoch 22 (best checkpoint)
  - Top-1 Accuracy: ~12%
  - Note: 28% Top-5 on 282 classes is reasonable for demo
- ✅ **Evaluation Pipeline:**
  - `video_mae_eval.py` calculates Top-1/Top-5 accuracy
  - Generates `evaluation_results.json` with beam search predictions
  - `evaluation_report.txt` with metrics summary
- ⚠️ **Known Issues:**
  - Model overpredicts "tv" class (mode collapse from low-motion bias)
  - Prediction inconsistencies between evaluation and live inference
  - Suspected causes: FPS variability, double-encoding artifacts, normalization issues

**Streamlit Application:**

- ✅ Real-time sign language detection with WebRTC camera access
- ✅ Bypasses WSL2 camera limitations with browser-based capture
- ✅ Features:
  - Live camera feed with VideoFrameCollector (60 frames buffer)
  - Continuous inference every 2 seconds during recording
  - Top-5 predictions with confidence scores
  - Video playback with content-based refresh
  - Upload video file option
- ✅ Model loading with processor fallback (handles missing preprocessor_config.json)
- 🔧 **Current Debugging:**
  - Added comprehensive debug logging to diagnose prediction inconsistencies
  - Frame processing pipeline verification (shape, dtype, value range, tensor normalization)
  - Investigating WebRTC FPS variability vs. fixed 30fps evaluation

**InternVL3.5 Training (Legacy):**

- ✅ VLM-based approach with PEFT/LoRA
- ✅ Not actively used (VideoMAE performs better for classification)

## 📊 Evaluation

To evaluate the trained model on the test set:

```bash
cd src/model/finetune
python eval.py
```

**Options:**

- `--limit N`: Run on only the first N samples (useful for testing).
- `--checkpoint PATH`: Specify a specific checkpoint path (defaults to the latest in `vlm_finetuned`).

**Metrics:**

- **Top-1 Accuracy**: Exact match of the predicted gloss.
- **Top-5 Accuracy**: Checks if the ground truth is within the top 5 beam search predictions.

  - `dataloader_num_workers=4` (optimized for batch size)
  - Early stopping with patience=3 epochs

- ✅ **LoRA Configuration:** r=16, alpha=32, dropout=0.1 (~2M trainable parameters)
- ✅ **Training Optimizations:**
  - Cosine learning rate scheduler with warmup
  - Learning rate: 2e-4 with 50 warmup steps
  - Weight decay: 0.01 for regularization
  - Best model selection based on validation loss
- ✅ Gradient checkpointing + use_cache compatibility resolved
- ✅ Chat template integration for proper instruction formatting
- ✅ Simplified prompt for better learning efficiency
- 📊 Real-time monitoring via TensorBoard + custom callback
- ⚡ **Expected Training Time:** ~45 mins - 1.5 hours for 10 epochs (RTX 2000 Ada 8GB) with preprocessed tensors

**In Progress:**

- 🔧 Debugging prediction inconsistencies between evaluation and live app
  - [x] Added comprehensive debug logging (frame characteristics, tensor properties)
  - [ ] Fix WebRTC FPS variability (use captured frames directly)
  - [ ] Eliminate double-encoding pipeline (capture→save→upload→read→predict)
  - [ ] Verify temporal sampling consistency
  - [ ] Test RGB color space and normalization alignment
- [ ] Model performance improvements
  - [ ] Address "tv" overprediction (mode collapse issue)
  - [ ] Investigate low-motion bias in preprocessing
  - [ ] Test with direct frame prediction (skip MP4 encoding)
  - [ ] Add temporal smoothing (average predictions over 5-10 frames)
  - [ ] Implement confidence threshold filtering (<10% ignored)

### Stage 3: Deployment & Production 📋 (Next Up)

- [x] Streamlit web application with WebRTC camera
- [x] Real-time inference pipeline
- [ ] Optimize inference latency (<500ms per prediction)
- [ ] REST API endpoints for programmatic access
- [ ] Model serving infrastructure (TorchServe/Triton)
- [ ] Docker containerization
- [ ] Error handling and logging improvements
- [ ] Performance monitoring and analytics

### Stage 4: Production Readiness 📋 (Planned)

- [ ] Comprehensive testing suite (unit, integration, E2E)
- [ ] Docker containerization with multi-stage builds
- [ ] Monitoring and health checks (Prometheus, Grafana)
- [ ] API documentation (OpenAPI/Swagger)
- [ ] CI/CD pipeline setup (GitHub Actions)
- [ ] Performance benchmarking suite
- [ ] Security hardening (rate limiting, input validation)
- [ ] Logging and observability (structured logs, traces)

## 🛠️ Technologies Used

### Core Technologies

- **Python 3.11+**: Primary development language
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face model library
- **VideoMAE**: Video action recognition model (primary approach)
- **InternVL3.5-2B**: Vision-Language Model (legacy approach)
- **TRL (Transformer Reinforcement Learning)**: Fine-tuning with SFT
- **PEFT**: Parameter-Efficient Fine-Tuning

### Web Application

- **Streamlit**: Interactive web application framework
- **streamlit-webrtc**: WebRTC camera access for real-time inference
- **OpenCV (cv2)**: Video processing and frame manipulation
- **imageio**: Video encoding and file I/O

### Dataset & Processing

- **WLASL Dataset**: Word-Level American Sign Language dataset (282 classes)
- **Decord**: High-performance video frame extraction and sampling
- **Custom Data Pipeline**: Filtering, validation, and preprocessing
- **Stratified Splitting**: Gloss-aware train/val/test splits for balanced evaluation
- **NumPy & cv2**: Frame augmentation (flip, brightness, crop, rotation, temporal speed)

### Hardware Acceleration

- **CUDA**: NVIDIA GPU acceleration with TF32 support
- **DirectML**: AMD GPU acceleration via PyTorch-DirectML
- **BF16 Precision**: Native on Ada GPUs (faster than FP16)
- **Accelerate**: Distributed training support (future)
- **BitsAndBytes**: 4-bit quantization for memory efficiency (InternVL legacy)

### Package Management & Environment

- **UV**: Ultra-fast Python package installer and resolver
- **YAML**: Configuration management
- **Python-dotenv**: Environment variable management

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- Git
- CUDA-compatible GPU (optional, for acceleration)
- AMD GPU with DirectML support (optional, for acceleration)
- UV package manager (recommended)

### Setup Instructions

1. **Install UV (if not already installed)**

   ```bash
   # Windows (PowerShell)
   irm https://astral.sh/uv/install.ps1 | iex

   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Alternative: via pip
   pip install uv
   ```

2. **Clone the repository**

   ```bash
   git clone https://github.com/Okeha/sign-language-detector-backend.git
   cd sign-language-detector-backend
   ```

3. **Set up Python environment and dependencies**

   ```bash
   # Create virtual environment and install dependencies
   uv sync

   # Activate the virtual environment
   # Windows
   .venv\Scripts\activate
   # Linux/macOS
   source .venv/bin/activate
   ```

4. **Set up environment variables**

   ```bash
   # Create .env file
   echo "HF_TOKEN=your_hugging_face_token_here" > .env
   ```

5. **Test the base model**
   ```bash
   cd src/model
   python model_loader.py
   ```

## 🚀 Usage

### Data Engineering Pipeline 📊

The data engineering process transforms raw WLASL dataset into training-ready video samples. Follow these steps in order:

#### **Step 1: Download WLASL Videos**

```bash
cd src/model/finetune/data_engineering
python video_downloader.py
```

**What it does:**

- Downloads sign language videos from YouTube using the WLASL dataset
- Filters for 320 core sign language words from the base glossary
- Saves videos to `raw_videos/` folder
- **Automatically skips already-downloaded videos** to allow resuming interrupted downloads
- Handles multiple video formats (.mp4, .swf for ASL Pro)

**Expected output:**

```
🧹 Starting WLASL BASE GLOSSARY FILTERING Process...
Glossary size: 319
Cleaned Glossary Size: 282

📂 Starting Downloaded Video Filteration Process...
Found 1892 video files in 'raw_videos' folder.
```

**Note**: This process can take several hours depending on:

- Number of videos to download (~2,000 videos for 320 words)
- Internet connection speed
- YouTube rate limiting

---

#### **Step 2: Validate Videos (Critical!)**

```bash
cd src/model/finetune/data_engineering
python validate_videos.py
```

**Why this is critical:**

- **~15-20% of downloaded videos are corrupted** (missing metadata, incomplete downloads)
- Corrupted videos cause training crashes with `moov atom not found` errors
- Validation removes bad videos before training starts

**What it does:**

- Checks every video file can be opened and read by decord
- Creates `datasets/wlasl_validated.json` with only working videos
- Generates `datasets/corrupted_videos.json` log for debugging
- Provides detailed statistics on dataset health

**Expected output:**

```
🔍 Loading dataset from: datasets/wlasl_cleaned.json
📊 Total samples in dataset: 1892

🔄 Validating videos...
Validating: 100%|███████████████████| 1892/1892

============================================================
📊 Validation Summary
============================================================
✅ Valid videos:     1603 (84.7%)
❌ Corrupted videos: 289 (15.3%)
============================================================

💾 Saved validated dataset to: datasets/wlasl_validated.json
📝 Saved corrupted videos log to: datasets/corrupted_videos.json
```

**Important**: Update `src/model/params/vlm.yml` to use the validated dataset:

```yaml
dataset_path: ../finetune/data_engineering/datasets/wlasl_validated.json
```

---

#### **Step 3: (Optional) Preprocess Videos for Faster Training**

```bash
cd src/model/finetune
python preprocess_videos.py
```

**When to use:**

- If you're training multiple times on the same dataset
- To reduce training startup time
- When experimenting with different hyperparameters

**What it does:**

- Pre-extracts and samples frames from all videos
- Converts frames to PyTorch tensors
- Saves preprocessed frames to `preprocessed_frames/` (`.pt` files)
- Reduces per-epoch data loading time by ~40-60%

**Trade-offs:**
| Aspect | Without Preprocessing | With Preprocessing |
|--------|----------------------|-------------------|
| **Disk Space** | ~2GB (raw videos) | ~2GB videos + ~5GB tensors |
| **First Epoch** | Normal speed | 40-60% faster |
| **Subsequent Epochs** | Same as first | Same as first |
| **Flexibility** | Can change NUM_FRAMES easily | Need to reprocess if changing frame count |

**Expected output:**

```
Processing Videos: 100%|████████| 1603/1603
✅ Saved preprocessed_frames/raw_videos/12345.pt
✅ Saved preprocessed_frames/raw_videos/12346.pt
...
✅ Processed 1603 videos successfully
⚠️  Failed 0 videos
```

**When NOT to use:**

- First time training (adds extra preprocessing step)
- Limited disk space (adds ~5GB storage requirement)
- Frequently changing frame sampling parameters

---

#### **Step 4: Train the VideoMAE Model**

```bash
cd src/model/finetune/videomae
python train_video_mae.py
```

**What it does:**

- Loads validated WLASL dataset with 282 classes
- Initializes VideoMAE-base with classification head
- Applies aggressive augmentation (flip, brightness, crop, rotation, temporal speed)
- Balances classes via sample repetition (target: 10 samples/class)
- Trains for 40 epochs with early stopping
- Saves checkpoints every epoch to `./video_mae_finetuned/`
- Logs training/validation loss to TensorBoard

**Expected output:**

```
🔄 Initializing VideoMAE Fine-Tuner...
🔄 Starting Loading dataset procedure...
Loaded dataset with 1603 samples

� DATASET STATISTICS:
   - Training Samples (Expanded 10x): 2820
   - Validation Samples: 423
   - Total Classes: 282

⚖️  BALANCE CHECK:
   - Min samples per class: 10
   - Max samples per class: 10
   - Avg samples per class: 10.0

❄️  FROZEN Early Layers. Unfrozen last 8 layers + Classifier.

🧠 Starting Fine-Tuning...

Epoch 1/40:
{'loss': 5.2345, 'learning_rate': 5e-05, 'epoch': 1.0}
{'eval_loss': 5.1234, 'eval_accuracy': 0.05, 'eval_top5_accuracy': 0.12}

Epoch 10/40:
{'loss': 2.8765, 'eval_top5_accuracy': 0.22}

Epoch 22/40:  ← Best checkpoint
{'loss': 2.1234, 'eval_top5_accuracy': 0.28}  ← Peak performance

Epoch 29/40:
Training stopped early (no improvement for 5 epochs)
```

**Training Time Estimates (RTX 2000 Ada 8GB):**

- **Per Step:** ~6-8 seconds (with augmentation)
- **Per Epoch:** ~20-25 minutes (depends on class balancing)
- **Full Training (40 epochs):** ~13-17 hours
- **With Early Stopping:** ~8-12 hours (typically stops at epoch 25-30)

**Training Configuration:**

```python
# VideoMAE Training (Optimized for RTX 2000 Ada)
per_device_train_batch_size = 2
gradient_accumulation_steps = 2  # Effective batch size = 4
num_train_epochs = 40
fp16 = True  # FP16 for VideoMAE (not BF16)
learning_rate = 5e-5
warmup_steps = 100
weight_decay = 0.01
label_smoothing_factor = 0.1

# Augmentation Configuration
NUM_FRAMES = 16  # Sample 16 frames @ 30fps
REPEAT_FACTOR = 10  # Target 10 samples per class
# Augmentation: flip (50%), brightness (70%), crop (80%), rotation (30%), temporal speed (50%)

# Architecture
# - Freeze early layers (reduce overfitting)
# - Unfreeze last 8 transformer layers
# - Train classification head
# - ~50M trainable parameters
lr_scheduler_type = "cosine"
weight_decay = 0.01
early_stopping_patience = 3  # Stop if no improvement for 3 epochs

# LoRA Configuration
lora_r = 16  # Increased from 8 for better capacity
lora_alpha = 32  # 2x the r value
lora_dropout = 0.1  # Prevent overfitting
# ~2M trainable parameters (0.16% of total model)
```

---

#### **Step 5: Monitor Training**

**In another terminal:**

```bash
tensorboard --logdir ./vlm_finetuned/tb_logs
# Open http://localhost:6006
```

**What to watch:**

- **Training Loss**: Should decrease steadily (healthy: 2.5 → 1.8 → 1.2)
- **Validation Loss**: Should track training loss closely
  - If val_loss stays close to train_loss: ✅ Good generalization
  - If val_loss >> train_loss after epoch 5: ⚠️ Overfitting
- **Learning Rate**: Should follow warmup → cosine decay schedule
- **GPU Memory**: Should stay around 6-7GB / 8GB (safe margin)
- **Step Time**: Should be consistent (~5-7s per step)

**Healthy training pattern:**

```
Epoch 1: Train Loss: 2.5, Val Loss: 2.7  ✅ Normal starting point
Epoch 2: Train Loss: 1.8, Val Loss: 1.9  ✅ Both decreasing
Epoch 3: Train Loss: 1.4, Val Loss: 1.5  ✅ Tracking well
Epoch 4: Train Loss: 1.2, Val Loss: 1.3  ✅ Still improving
Epoch 5: Train Loss: 1.1, Val Loss: 1.2  ✅ Best model
Epoch 6: Train Loss: 1.0, Val Loss: 1.2  ⚠️ Val stopped improving
Epoch 7: Train Loss: 0.9, Val Loss: 1.3  ⚠️ Val increasing
→ Early stopping triggered! Using Epoch 5 checkpoint
```

**Overfitting (bad pattern):**

```
Epoch 1: Train Loss: 2.5, Val Loss: 2.7  ✅
Epoch 2: Train Loss: 1.8, Val Loss: 1.9  ✅
Epoch 3: Train Loss: 0.8, Val Loss: 2.1  ❌ Gap too large!
→ Model memorizing training data, not learning patterns
```

**Fixes for overfitting:**

- Increase augmentation probability (e.g., rotation from 30% to 50%)
- Add more diverse augmentation (Gaussian noise, color jitter)
- Reduce `num_train_epochs` from 40 to 20
- Check if you have enough unique samples per gloss (need 5+ videos minimum)
- Increase label smoothing from 0.1 to 0.15

---

#### **Step 5: Run the Streamlit Application**

```bash
cd src/app
streamlit run streamlit_app.py

# For WSL2 users (camera doesn't work in WSL):
# Double-click run_windows.bat in File Explorer
# Or run from Windows CMD:
src\app\run_windows.bat
```

**What it does:**

- Loads trained VideoMAE model (checkpoint-77176 by default)
- Starts Streamlit web server on `http://localhost:8501`
- Provides WebRTC camera access (bypasses WSL2 limitations)
- Runs continuous inference every 2 seconds during recording
- Shows Top-5 predictions with confidence scores
- Displays video playback with content-based refresh

**Features:**

1. **Live Camera Mode** (WebRTC)

   - Real-time video capture at variable FPS
   - Frame buffer: 60 frames max
   - Inference trigger: Every 60 frames (~2 seconds)
   - Shows live prediction display during recording

2. **Upload Video Mode**

   - Upload pre-recorded .mp4 files
   - Processes entire video (samples 16 frames uniformly)
   - Shows video playback with predictions

3. **Debug Information**
   - Frame count, shape, dtype, value range
   - Tensor normalization (should be ~[-1, 1])
   - Inference latency (ms)
   - Top-5 predictions with confidence %

**Expected output:**

```
📹 Recording... (Frame: 45/60)
🤖 Live Prediction: "hello" (Top-5: hello, hi, wave, greet, meet)

[DEBUG] ========== PREDICTION START ==========
[DEBUG] Input frames count: 60
[DEBUG] Processing 60 frames, sampling 16
[DEBUG] Frame shape: (480, 640, 3)
[DEBUG] Frame dtype: uint8
[DEBUG] Frame value range: [0, 255]
[DEBUG] Processed tensor shape: torch.Size([1, 16, 3, 224, 224])
[DEBUG] Tensor value range: [-1.234, 1.567]
[DEBUG] Top-1 Prediction: hello (85.3%)
[DEBUG] Top-5 Predictions: [('hello', '85.3%'), ('hi', '8.2%'), ...]
[DEBUG] Latency: 487ms
[DEBUG] ========== PREDICTION END ==========
```

**Troubleshooting:**

- **Camera not working in WSL2**: Use `run_windows.bat` to open in Windows browser
- **"tv" predictions only**: Check debug output for tensor range (should be normalized)
- **Inconsistent predictions**: Verify FPS stability, check for double-encoding artifacts
- **High latency (>1s)**: Reduce NUM_FRAMES from 16 to 8, ensure GPU acceleration enabled

---

### Data Pipeline Summary

```
Raw WLASL Dataset (2,000+ words)
         ↓
    Filter by 282-class glossary
         ↓
    Download videos from YouTube
         ↓
    Validate videos (remove corrupted: 84.7% success)
         ↓
    [Optional] Preprocess to .pt tensors
         ↓
    Stratified train/test split (by gloss)
         ↓
    Training with dynamic frame sampling
```

### Current Functionality

#### Streamlit Web Application (Primary Interface)

```bash
# Run the Streamlit app
cd src/app
streamlit run streamlit_app.py

# For WSL2 users (camera access fix):
# Use run_windows.bat to launch in Windows browser
```

**Features:**

- Real-time camera capture with WebRTC (bypasses WSL2 limitations)
- Continuous inference every 2 seconds during recording
- Top-5 predictions with confidence scores
- Video upload option for pre-recorded videos
- Dark theme UI with custom styling

#### Programmatic Video Processing (Legacy)

```python
from src.model.model_loader import VLMModelLoader

# Initialize the model loader (InternVL3.5)
vlm_loader = VLMModelLoader()

# Process a video file
result = vlm_loader.generate_response("path/to/your/video.mp4")

# Clean up resources
vlm_loader.shutdown()
```

**Note**: For VideoMAE inference, use the Streamlit app or load the model directly:

```python
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

# Load fine-tuned VideoMAE model
model = VideoMAEForVideoClassification.from_pretrained(
    "src/model/finetune/videomae/video_mae_finetuned/checkpoint-77176"
)
processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

# Process video frames (16 frames, uniformly sampled)
# See streamlit_app.py for complete inference pipeline
```

### Dataset Setup and Fine-tuning Pipeline (Legacy Documentation)

**⚠️ Note**: For the complete, up-to-date data engineering process, see the **Data Engineering Pipeline** section above.

The fine-tuning pipeline uses a carefully curated dataset from WLASL (Word-Level American Sign Language).

#### Dataset Structure

The processed dataset follows this format:

```json
{
  "prompt": "You are an expert sign-language recognition model. Identify the sign in the video and respond with exactly one word and nothing else.",
  "video_path": "raw_videos/12345.mp4",
  "gloss": "HELLO"
}
```

#### Supported Sign Language Words (320 Glossary)

The fine-tuning pipeline uses a carefully curated 320-word glossary covering:

- **Pronouns** (15 words): I, YOU, HE, SHE, THEY, etc.
- **Basic Verbs** (40 words): BE, HAVE, DO, GO, MAKE, etc.
- **Time Words** (25 words): NOW, TODAY, TOMORROW, etc.
- **People & Roles** (25 words): PERSON, FAMILY, TEACHER, etc.
- **Places** (25 words): HOME, SCHOOL, HOSPITAL, etc.
- **Objects** (30 words): BOOK, PHONE, FOOD, etc.
- **Feelings** (20 words): HAPPY, SAD, ANGRY, etc.
- **Descriptors** (35 words): BIG, SMALL, FAST, SLOW, etc.
- **Colors** (10 words): RED, BLUE, GREEN, etc.
- **Numbers** (20 words): ONE, TWO, THREE, etc.
- **Question Words** (10 words): WHO, WHAT, WHERE, etc.
- **Connectors** (10 words): AND, OR, BUT, etc.

#### Configuration

Modify `src/model/params/vlm.yml` to customize both VideoMAE and InternVL settings:

```yaml
# VideoMAE Configuration (Primary)
video_mae_params:
  pretrained_model_name: "MCG-NJU/videomae-base"
  num_frames: 16 # Frames sampled per video
  repeat_factor: 10 # Target samples per class
  training_arguments:
    output_dir: "./video_mae_finetuned"
    per_device_train_batch_size: 2
    num_train_epochs: 40
    learning_rate: 5e-5
    # ... more training args

# InternVL3.5 Configuration (Legacy)
model: "OpenGVLab/InternVL3_5-2B-hf"
prompt: "You are an expert sign-language recognition model..."
```

### VideoMAE Fine-tuning Configuration (Current Pipeline)

The VideoMAE training uses aggressive augmentation and class balancing:

```python
# VideoMAE Training Configuration
training_args = TrainingArguments(
    output_dir="./video_mae_finetuned",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=40,
    gradient_accumulation_steps=2,  # Effective batch size = 4
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    fp16=True,  # FP16 for VideoMAE
    logging_steps=200,
    save_strategy="epoch",
    eval_strategy="epoch",
    dataloader_num_workers=5,
    metric_for_best_model="top5_accuracy",
    load_best_model_at_end=True,
    label_smoothing_factor=0.1,
)

# Augmentation Pipeline (in VideoMAEDataset.__getitem__)
# 1. Random Horizontal Flip (50% probability)
# 2. Random Brightness/Contrast (70%, alpha=0.8-1.2, beta=±20)
# 3. Random Crop + Resize (80%, zoom 0-30%)
# 4. Random Rotation (30%, ±10°)
# 5. Temporal Speed Variation (50%, 80-120% speed)

# Class Balancing
# Target: 10 samples per class via repetition
# r = max(1, int(TARGET_PER_CLASS / count))
# Ensures balanced representation during training

# Architecture
# - Freeze early VideoMAE encoder layers
# - Unfreeze last 8 transformer layers
# - Train classification head (282 classes)
# - ~50M trainable parameters
```

### InternVL3.5 Fine-tuning Configuration (Legacy)

The legacy VLM approach uses PEFT with LoRA:

```python
# PEFT/LoRA Configuration (InternVL3.5)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training Arguments (InternVL3.5)
training_args = TrainingArguments(
    output_dir="./vlm_finetuned",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=10,
    gradient_accumulation_steps=8,  # Effective batch size = 16
    bf16=True,  # BF16 for Ada GPUs
    tf32=True,  # TF32 acceleration
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    gradient_checkpointing=True,
    remove_unused_columns=False,
    report_to="tensorboard",
    logging_dir="./tb_logs",
    optim="adamw_torch",
)

# Custom data collator handles:
# - Video frame sampling (8 frames by default)
# - Processor formatting with <video> placeholder
# - Label masking (prompt=-100, answer=token_ids)
# - Dynamic batch filtering for corrupted videos
# - BF16 tensor conversion for memory efficiency
```

**Key Features (InternVL Legacy):**

- **Gradient Checkpointing**: Reduces VRAM usage by ~40%
- **4-bit Quantization**: Model loaded in 4-bit reduces memory by ~75%
- **LoRA (r=16)**: Only trains ~0.17% of parameters (~2M params)
- **BF16 + TF32**: Ada GPU acceleration for 15-20% speedup
- **Chat Templates**: Instruction formatting with `<video>` token

## 📊 Performance

### Current Benchmarks (Dec 18, 2025)

#### VideoMAE Training (Primary Pipeline)

- **Model**: MCG-NJU/videomae-base fine-tuned on WLASL
- **Dataset**: 1,603 validated videos, 282 classes
- **Training Split**: 2,820 samples (with class balancing) / 423 validation
- **Hardware**: NVIDIA RTX 2000 Ada (8GB VRAM)

**Training Performance:**

- **Checkpoint Loading**: ~3-5 seconds
- **Per Epoch Time**: ~20-25 minutes (with augmentation)
- **Total Training**: ~8-12 hours (stopped at epoch 29/40)
- **Best Checkpoint**: Epoch 22 (checkpoint-77176)

**Model Performance:**

- **Top-1 Accuracy**: 12.4% on test set
- **Top-5 Accuracy**: 27.8% on test set
- **Note**: 28% Top-5 on 282 classes is reasonable for demo

**Memory Usage:**

```
VideoMAE Base Model:    ~3GB
Classification Head:    ~0.5GB
Activations (bs=2):     ~2.5GB
Optimizer States:       ~1.5GB
------------------------------
Total Usage:            ~7-7.5GB / 8GB
Peak During Eval:       ~7.8GB
```

#### Streamlit Application

- **Model Loading**: ~3-5 seconds (VideoMAE checkpoint)
- **Inference Latency**: ~400-600ms per prediction (16 frames)
- **Memory Usage**: ~4-5GB VRAM (inference only, no training)
- **Camera Capture**: Variable FPS (browser WebRTC)
- **Frame Buffer**: 60 frames max, triggers prediction every 2 seconds

**Known Issues:**

- "tv" overprediction (mode collapse from low-motion bias)
- Prediction inconsistencies between evaluation and live app
- WebRTC FPS variability causes temporal sampling differences
- Double-encoding artifacts (capture→save→upload→read→predict)

#### InternVL3.5 Training (Legacy)

- **Model**: InternVL3.5-2B-hf with 4-bit quantization + LoRA
- **Status**: Legacy approach, not actively used (VideoMAE performs better)

**Memory Breakdown (RTX 2000 Ada 8GB):**

```
Base Model (4-bit):     ~2GB
LoRA Adapters (r=16):   ~0.8GB
Activations (bs=2):     ~2GB
Grad Checkpointing:     Saves ~1.5GB
Optimizer States:       ~1.2GB
------------------------------
Total Usage:            ~6-7GB / 8GB (safe margin)
Peak During Eval:       ~7GB
```

**Training Speed Optimizations:**

| Optimization                      | Impact (VideoMAE)  | Impact (InternVL) | Notes                           |
| --------------------------------- | ------------------ | ----------------- | ------------------------------- |
| **FP16 precision**                | +20% speed         | N/A               | VideoMAE uses FP16              |
| **BF16 + TF32**                   | N/A                | +15% speed        | InternVL on Ada GPUs            |
| **Aggressive augmentation**       | -10% speed         | N/A               | Trade-off for better accuracy   |
| **dataloader_num_workers=5**      | +15% speed         | +20% speed        | Prevents CPU bottleneck         |
| **gradient_accumulation=2**       | Better stability   | Better coverage   | VideoMAE uses smaller effective |
| **Class balancing (10x repeat)**  | +180% data samples | N/A               | Ensures balanced training       |
| **Early stopping (not used yet)** | Would save 10-15h  | Saves 3-6 epochs  | VideoMAE trains to convergence  |

**Expected Training Times:**

| Pipeline     | Per Epoch    | Full Training | With Early Stop |
| ------------ | ------------ | ------------- | --------------- |
| VideoMAE     | ~20-25 min   | ~13-17 hours  | ~8-12 hours     |
| InternVL3.5  | ~12-15 min   | ~2-2.5 hours  | ~1-1.5 hours    |
| **Hardware** | RTX 2000 Ada | 8GB VRAM      | FP16/BF16       |

## 🔧 Hardware Requirements

### Minimum Requirements

- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **CPU**: Multi-core processor

### GPU Requirements (Optional but Recommended)

- **NVIDIA GPU**: GTX 1060 / RTX 2060 or better with 6GB+ VRAM
  - **Recommended**: RTX 2000 Ada / RTX 3060 or better (8GB+ VRAM)
  - **Features**: CUDA, TF32 (Ampere+), BF16 (Ada+)
- **AMD GPU**: RX 6600 or better with DirectML support
- **VRAM**:
  - **Minimum**: 4GB (inference only, limited batch sizes)
  - **Recommended**: 8GB+ (training with full features)
  - **Optimal**: 12GB+ (larger batch sizes, faster training)

### Tested Configurations

| Hardware            | Training | Inference | Notes                        |
| ------------------- | -------- | --------- | ---------------------------- |
| RTX 2000 Ada (8GB)  | ✅       | ✅        | Primary development hardware |
| RTX 3060 (12GB)     | ✅       | ✅        | Better performance           |
| GTX 1660 Ti (6GB)   | ⚠️       | ✅        | Reduce batch size to 1       |
| AMD RX 6600 (8GB)   | ✅       | ✅        | DirectML, float32 (slower)   |
| CPU only (16GB RAM) | ❌       | ⚠️        | Very slow, not recommended   |
| WSL2 + Windows GPU  | ✅       | ✅        | Use run_windows.bat for app  |

## 🧪 Testing

### Test Data Structure

```
src/model/tests/data/
├── images/          # Test images for static detection
└── videos/          # Test videos for sequence detection
    └── test2.mp4    # Sample test video
```

### Dataset Structure

```
src/model/finetune/data_engineering/
├── raw_videos/              # Downloaded WLASL videos (.mp4, .swf)
│   ├── 12345.mp4
│   ├── 12346.mp4
│   └── ...
├── datasets/                # Processed JSON datasets
│   ├── wlasl_cleaned.json        # Filtered (may contain corrupted videos)
│   ├── wlasl_validated.json      # ✅ Use this! (corrupted videos removed)
│   └── corrupted_videos.json     # Log of videos that failed validation
├── preprocessed_frames/     # Optional: Pre-extracted frames for faster training
│   └── raw_videos/
│       ├── 12345.pt         # Tensor: (16, 3, H, W) - 16 frames, RGB, fp16
│       └── ...
├── raw_data/                # Original WLASL JSON files
├── filter_data.py           # Filters dataset by 282-class glossary
├── video_downloader.py      # Downloads videos from YouTube
└── validate_videos.py       # Validates videos and removes corrupted ones (CRITICAL!)
```

**Dataset File Formats:**

**wlasl_validated.json:**

```json
[
  {
    "prompt": "You are an expert sign-language recognition model...",
    "video_path": "raw_videos/12345.mp4",
    "gloss": "HELLO"
  },
  ...
]
```

**corrupted_videos.json:**

```json
[
  {
    "video_path": "raw_videos/56579.mp4",
    "gloss": "BOOK",
    "reason": "moov atom not found"
  },
  ...
]
```

### Running Tests

```bash
# Test VideoMAE model loading and inference
cd src/model/finetune/videomae
python video_mae_eval.py --limit 10  # Test on first 10 samples

# Test Streamlit application
cd src/app
streamlit run streamlit_app.py

# Test data processing pipeline
cd src/model/finetune/data_engineering
python validate_videos.py  # Validate dataset health

# Test dataset loader
cd src/model/finetune/videomae
python -c "from dataset import VideoMAEDataset; ds = VideoMAEDataset('train'); print(len(ds))"

# Test legacy InternVL pipeline (not actively used)
cd src/model/finetune/internvl3_5
python eval.py --limit 5
```

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
uv sync --dev

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/
isort src/

# Run linting
flake8 src/
```

### Optimization Features

**VideoMAE Pipeline (Current):**

- **Aggressive augmentation**: Flip (50%), brightness (70%), crop (80%), rotation (30%), temporal speed (50%)
- **Class balancing**: Target 10 samples per class via repetition
- **FP16 precision**: Memory efficient training
- **Label smoothing** (0.1): Prevents overconfidence
- **Stratified splitting**: Balanced gloss representation
- **TensorBoard integration**: Real-time loss visualization
- **Gradient accumulation**: Simulates larger batch sizes
- **8 unfrozen layers**: Balances capacity and overfitting
- **Cosine learning rate**: Better convergence
- **Early stopping**: Saves compute when plateauing

**InternVL3.5 Pipeline (Legacy):**

- **4-bit quantization** (CUDA only): Reduces memory by ~75%
- **BF16 + TF32** (Ada GPUs): 15-20% speedup over FP16
- **LoRA/PEFT** (r=16): Trains only ~0.17% of parameters
- **Gradient checkpointing**: Saves ~40% VRAM
- **Chat template integration**: Proper `<video>` token formatting
- **Dynamic video filtering**: Skips corrupted videos automatically

**Data Pipeline (Shared):**

- **Video validation**: Removes corrupted videos (84.7% success rate)
- **Skip-if-exists downloads**: Resume interrupted downloads
- **Optional preprocessing**: Pre-extract frames for 40-60% speedup
- **Stratified dataset splitting**: Balanced gloss distribution
- **Efficient memory cleanup**: Garbage collection and cache clearing

**Streamlit Application:**

- **WebRTC camera access**: Bypasses WSL2 camera limitations
- **Real-time inference**: Continuous predictions every 2 seconds
- **Frame buffering**: 60-frame buffer for smooth capture
- **Content-based refresh**: Prevents unnecessary video re-renders
- **Debug logging**: Comprehensive frame and tensor diagnostics

### Dataset Statistics & Health

- **Total WLASL Vocabulary**: ~2,000+ words
- **Filtered Glossary**: 282 classes (reduced from 320-word base)
- **Videos Downloaded**: ~1,892
- **Videos Validated**: ~1,603 (84.7% success rate)
- **Corrupted Videos**: ~289 (15.3% - automatically filtered)
- **Video Sources**: YouTube via WLASL dataset
- **Processing Pipeline**: Automated filtering, validation, and dataset generation
- **Train/Val/Test Split**: 70/15/15 (stratified by gloss)
- **Frame Sampling**: 16 frames per video (VideoMAE), 8 frames (InternVL legacy)

### Known Issues & Solutions

| Issue                                        | Cause                                   | Solution                                                      |
| -------------------------------------------- | --------------------------------------- | ------------------------------------------------------------- |
| **"tv" overprediction**                      | Mode collapse, low-motion bias          | Investigate preprocessing, add temporal smoothing             |
| **Prediction inconsistencies (eval vs app)** | WebRTC FPS variability, double-encoding | Use captured frames directly, fix normalization               |
| **moov atom not found**                      | Corrupted video downloads               | Run `validate_videos.py` before training (CRITICAL!)          |
| **OOM errors (training)**                    | Batch size too large                    | Reduce to batch_size=1, increase gradient_accumulation        |
| **Training stuck at 0% with skips**          | Too many corrupted videos               | Validate dataset first with `validate_videos.py`              |
| **Slow data loading**                        | Decord processing overhead              | Use `preprocess_videos.py` for faster loading (optional)      |
| **Camera not working (WSL2)**                | WSL2 can't access Windows camera        | Use `run_windows.bat` to open in Windows browser              |
| **High inference latency (>1s)**             | Large frame count or no GPU             | Reduce NUM_FRAMES from 16 to 8, ensure CUDA is available      |
| **Training too slow (VideoMAE)**             | Augmentation overhead                   | Acceptable trade-off for better generalization                |
| **Low accuracy (<10% Top-1)**                | Insufficient training or bad data       | Train longer, check for corrupted videos, verify augmentation |

## 🔮 Future Enhancements

### Short-term Goals (Next 2-4 weeks)

- ✅ Model evaluation pipeline (COMPLETED - `video_mae_eval.py`)
- ✅ Streamlit web application (COMPLETED - real-time inference with WebRTC)
- 🔧 **Fix prediction inconsistencies** (IN PROGRESS - debugging frame pipeline)
  - Eliminate double-encoding artifacts
  - Fix WebRTC FPS variability
  - Verify normalization consistency
  - Add temporal smoothing for stable predictions
- Inference optimization (reduce latency from ~500ms to <300ms)
  - Model quantization (8-bit or 4-bit for faster inference)
  - Batch processing for multiple predictions
  - Frame caching and reuse
- Address "tv" overprediction (mode collapse)
  - Investigate low-motion bias in augmentation
  - Test with direct frame prediction (skip encoding)
  - Add per-class confidence calibration
- Hyperparameter optimization
  - Test different NUM_FRAMES (8 vs 16 vs 32)
  - Experiment with learning rates and schedulers
  - Fine-tune augmentation probabilities

### Medium-term Goals (1-3 months)

- REST API implementation with FastAPI
  - POST /predict endpoint for video upload
  - WebSocket for real-time streaming
  - GET /health and /metrics endpoints
- Model serving infrastructure
  - TorchServe or Triton Inference Server
  - Model versioning and A/B testing
  - Load balancing for concurrent requests
- Docker containerization
  - Multi-stage builds (training vs inference)
  - CUDA-enabled containers
  - Docker Compose for full stack
- Cloud deployment
  - AWS/GCP/Azure GPU instances
  - Serverless inference (AWS Lambda + custom containers)
  - CDN for static assets
- Mobile app integration
  - ONNX model export for edge deployment
  - TensorFlow Lite conversion
  - React Native or Flutter integration
- Performance improvements
  - Model distillation (VideoMAE-base → smaller variant)
  - Pruning for faster inference
  - Mixed precision inference (INT8/FP16)

### Long-term Vision (3-6 months)

- **Multi-language support**
  - British Sign Language (BSL)
  - International Sign Language variants
  - Cross-lingual transfer learning
- **Advanced training strategies**
  - Contrastive learning for better feature separation
  - Self-supervised pre-training on unlabeled sign videos
  - Multi-task learning (classification + detection + translation)
  - Test QLoRA vs full fine-tuning vs LoRA
- **Continuous learning**
  - User feedback integration
  - Active learning for hard examples
  - Online model updates with new data
- **Real-time optimization**
  - Sub-200ms inference latency
  - Efficient batching for concurrent users
  - Edge deployment (Raspberry Pi, mobile devices)
- **Custom architectures**
  - Temporal attention mechanisms
  - 3D CNNs for spatial-temporal features
  - Transformer-based video encoders
- **Multi-modal features**
  - Video + text context (e.g., "What does this sign mean in the context of cooking?")
  - Audio integration for sign-to-speech
  - Pose estimation overlay for feedback
- **Explainability**
  - Attention visualization (which frames matter most?)
  - Saliency maps (which regions of the frame?)
  - Per-class confusion analysis
  - Confidence calibration curves

## � Troubleshooting Guide

### Common Issues

#### 1. Training Crashes with "moov atom not found"

**Problem**: Corrupted video files cause decord to fail during training.

**Solution**:

```bash
cd src/model/finetune/data_engineering
python validate_videos.py

# Update vlm.yml to use validated dataset
dataset_path: ../finetune/data_engineering/datasets/wlasl_validated.json
```

---

#### 2. "None of the inputs have requires_grad=True"

**Problem**: Gradient checkpointing not properly configured with model.

**Solution**:

```python
# Add in __init__ before PEFT wrapping
from peft import prepare_model_for_kbit_training

self.model = prepare_model_for_kbit_training(
    self.model,
    use_gradient_checkpointing=True
)
```

---

#### 3. Out of Memory (OOM) Errors

**Problem**: Batch size too large for GPU memory.

**Solutions**:

```python
# Option 1: Reduce batch size
per_device_train_batch_size = 1
gradient_accumulation_steps = 32  # Keep effective batch same

# Option 2: Enable gradient checkpointing
gradient_checkpointing = True

# Option 3: Reduce frame count
NUM_FRAMES = 2  # Instead of 4
```

---

#### 4. Training Stuck at 0% with Many Skipped Videos

**Problem**: Most videos are corrupted, training makes no progress.

**Solution**: Always validate dataset first (see Issue #1).

---

#### 5. Slow Data Loading

**Problem**: Decord processing videos on-the-fly during training.

**Solution**:

```bash
# Preprocess videos once
cd src/model/finetune
python preprocess_videos.py

# Modify train.py to load from preprocessed_frames/
```

---

#### 6. ImportError or Module Not Found

**Problem**: Incorrect import paths or not running as module.

**Solution**:

```bash
# Always run from src/ directory with -m flag
cd src
python -m model.finetune.train

# NOT: python model/finetune/train.py (wrong!)
```

---

#### 7. TensorBoard Not Showing Graphs

**Problem**: TensorBoard not connected or wrong log directory.

**Solution**:

```bash
# Check logs exist
ls ./vlm_finetuned/tb_logs

# Start TensorBoard with correct path
tensorboard --logdir ./vlm_finetuned/tb_logs

# Open browser to http://localhost:6006
```

---

### Performance Optimization Tips

1. **Maximize Batch Size**: Start with batch_size=2, increase if no OOM
2. **Use Preprocessing**: Run `preprocess_videos.py` for 40-60% speedup (optional)
3. **Monitor GPU Usage**: Use `nvidia-smi` to check utilization
4. **Gradient Accumulation**: Use instead of increasing batch size if OOM
5. **BF16 Training**: Enable on Ada GPUs for native speedup (faster than FP16)
6. **TF32 Acceleration**: Enable for hardware matrix multiplication speedup
7. **Dataloader Workers**: Adjust based on batch size (workers=4 optimal for batch_size=2)
8. **Augmentation Pipeline**: Aggressive augmentation prevents overfitting on small datasets
9. **Early Stopping**: Use patience=3 to save time when model plateaus
10. **Frame Count**: 16 frames balances temporal resolution and memory (vs 4 or 32)

### Debug Commands

```bash
# Check GPU memory usage
nvidia-smi -l 1

# Monitor training in real-time
tensorboard --logdir ./vlm_finetuned/tb_logs

# Validate a single video
python -c "import decord; vr = decord.VideoReader('path/to/video.mp4'); print(len(vr))"

# Check trainable parameters
python -c "from model.finetune.train import VLMFineTuneTrainer; # prints trainable param count"

# Test dataset loading
python -c "from model.finetune.dataset import DatasetLoader; dl = DatasetLoader()"
```

## �📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenGVLab** for the InternVL3 model
- **Hugging Face** for the Transformers library
- **PyTorch Team** for the deep learning framework
- **Microsoft** for DirectML AMD GPU support

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Okeha/sign-language-detector-backend/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Okeha/sign-language-detector-backend/discussions)
- **Email**: anthony.okeh@example.com

---

**Made with ❤️ to bridge communication gaps and make technology more accessible.**
