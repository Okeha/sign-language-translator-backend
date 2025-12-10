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
│   ├── model/                     # VLM Model Components
│   │   ├── __init__.py            # Package marker
│   │   ├── model_loader.py        # InternVL3 model loader and inference
│   │   ├── params/                # Model configuration
│   │   │   └── vlm.yml            # VLM parameters and prompts
│   │   ├── finetune/              # Fine-tuning pipeline (MAIN TRAINING PIPELINE)
│   │   │   ├── __init__.py        # Package marker
│   │   │   ├── data_engineering/  # Dataset processing
│   │   │   │   ├── __init__.py    # Package marker
│   │   │   │   ├── video_downloader.py     # YouTube video downloader
│   │   │   │   ├── filter_data.py          # WLASL data filtering
│   │   │   │   ├── raw_videos/             # Downloaded video files
│   │   │   │   └── datasets/               # Processed datasets (JSON)
│   │   │   ├── dataset.py         # Dataset loader with stratified split
│   │   │   ├── train.py           # Fine-tuning orchestrator
│   │   │   └── eval.py            # Model evaluation (planned)
│   │   └── tests/                 # Test data and validation
│   │       └── data/
│   │           ├── images/        # Test images
│   │           └── videos/        # Test videos
│   └── api/                       # REST API endpoints (planned)
├── main.py                        # Application entry point
├── pyproject.toml                 # UV package dependencies
└── README.md                      # This file
```

### Package Structure Notes

- All directories have `__init__.py` files for proper Python package imports
- Use **relative imports** throughout the codebase
- Run modules from the `src/` package root: `python -m model.finetune.train`
- The `finetune/` directory is the **canonical training pipeline** (no duplicate train folders)

## 🚀 Development Stages

### Stage 1: Foundation ✅ (Completed)

- [x] Project structure setup
- [x] VLM model integration (InternVL3-2B → InternVL3.5-2B)
- [x] Multi-hardware support (CUDA/DirectML/CPU)
- [x] Basic video processing capabilities
- [x] Configuration management
- [x] Memory optimization and cleanup
- [x] UV package management setup

### Stage 2: Data Pipeline & Fine-tuning 🚧 (In Progress - 95% Complete)

**Completed:**

- [x] WLASL dataset integration (2,000+ words)
- [x] YouTube video downloader with skip-if-exists optimization
- [x] Data filtering pipeline (320-word core glossary)
- [x] Dataset validation and corruption detection
- [x] Dataset generation for fine-tuning
- [x] Dataset loader with gloss-aware stratified train/val/test split (70/15/15 by gloss)
- [x] Training pipeline with SFT configuration
- [x] Package structure with proper `__init__.py` files and relative imports
- [x] Custom video collation function with dynamic frame sampling
- [x] Label masking implementation (prompt tokens = -100, answer tokens = target)
- [x] PEFT/LoRA integration with gradient checkpointing
- [x] Complete training loop with Hugging Face Trainer
- [x] Hardware-specific optimizations (fp16 for CUDA, float32 for DirectML)
- [x] TensorBoard integration for real-time loss monitoring
- [x] Batch size optimization and gradient accumulation
- [x] Video preprocessing script for faster training (optional)
- [x] Evaluation pipeline with Top-1/Top-5 Accuracy metrics

**Current Status (as of Nov 27, 2025):**

- ✅ Successfully trained on 1,180 train samples / 423 test samples
- ✅ Model: InternVL3.5-2B-hf with 4-bit quantization (CUDA)
- ✅ **Optimized Training Configuration:**
  - `batch_size=2`, `gradient_accumulation_steps=8` (effective batch=16)
  - `bf16=True` (native for Ada GPUs), `tf32=True` (hardware acceleration)
  - `dataloader_num_workers=4` (faster data loading)
  - **8 Frames per Video** (Increased from 4 for better temporal resolution)
  - **Data Augmentation** (ColorJitter)
  - **Label Smoothing** (0.1)
  - **Train/Val/Test Split** (70/15/15)
- ✅ **Evaluation Pipeline:**
  - `eval.py` script calculates Accuracy and Top-5 Accuracy
  - Generates `evaluation_results.json` with detailed predictions

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

- [x] Gradient checkpointing + use_cache configuration (RESOLVED)
- [x] LoRA hyperparameter optimization (r=16, alpha=32)
- [x] Training speed optimizations (bf16, tf32, reduced workers)
- [x] Prompt engineering for better instruction following
- [x] Model evaluation pipeline on test set with accuracy metrics
- [x] Checkpoint management and best model selection
- [ ] Inference optimization for deployed model
- [x] Testing with different NUM_FRAMES values (4 vs 8 vs 16)

### Stage 3: Model Evaluation & Deployment 📋 (Next Up)

- [ ] REST API endpoints for image/video processing
- [ ] WebSocket integration for real-time streaming
- [ ] Request/response validation
- [ ] Error handling and logging
- [ ] Authentication and rate limiting

### Stage 4: Production Readiness 📋 (Planned)

- [ ] Docker containerization
- [ ] Monitoring and health checks
- [ ] Comprehensive testing suite
- [ ] Documentation and API references
- [ ] CI/CD pipeline setup

## 🛠️ Technologies Used

### Core Technologies

- **Python 3.11+**: Primary development language
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face model library
- **InternVL3-2B**: Vision-Language Model for sign language interpretation
- **TRL (Transformer Reinforcement Learning)**: Fine-tuning with SFT
- **PEFT**: Parameter-Efficient Fine-Tuning

### Dataset & Processing

- **WLASL Dataset**: Word-Level American Sign Language dataset
- **Decord**: High-performance video frame extraction and sampling
- **Custom Data Pipeline**: Filtering and preprocessing for 320-word glossary
- **Stratified Splitting**: Gloss-aware train/test splits for balanced evaluation

### Hardware Acceleration

- **CUDA**: NVIDIA GPU acceleration
- **DirectML**: AMD GPU acceleration via PyTorch-DirectML
- **BitsAndBytes**: 4-bit quantization for memory efficiency
- **Accelerate**: Distributed training support

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

#### **Step 4: Train the Model**

```bash
cd src/model/finetune
python -m train
```

**What it does:**

- Loads validated dataset with stratified train/test split
- Initializes InternVL3.5-2B with 4-bit quantization
- Applies LoRA adapters for parameter-efficient fine-tuning
- Trains for 3 epochs with gradient accumulation
- Saves checkpoints every epoch to `./vlm_finetuned/`
- Logs training/validation loss to TensorBoard

**Expected output:**

```
✅ Found NVIDIA GPU: NVIDIA RTX 2000 Ada Generation Laptop GPU
🚀 Starting OpenGVLab/InternVL3_5-2B-hf VLM Model Loading...
--- Loading for CUDA in float16 ---
✅  4-bit quantization is ENABLED for CUDA
✅ Model Loaded Successfully in ⏱️: 8.67 seconds.

🔄 Starting Loading dataset procedure...
Loaded dataset with 1603 samples

🔄 Starting Train Test Split...
Total: 1180 train, 423 test samples
Split ratio: 73.6% train

🔍 Trainable params: 2,097,152 / 1,234,567,890
   Percentage: 0.17%

🧠 Starting Fine-Tuning...

Epoch 1/10:
[step 10] loss = 2.4567
[step 20] loss = 2.1234
...
Epoch 1: 100%|████| 148/148 [12:34<00:00, 0.20it/s, loss=1.8765]
Eval Loss: 1.9123

Epoch 2/10:
...
```

**Training Time Estimates (RTX 2000 Ada 8GB):**

- **Per Step:** ~5-7 seconds
- **Per Epoch:** ~12-15 minutes (148 steps with batch_size=2, grad_accum=8)
- **Full Training (10 epochs):** ~2-2.5 hours
- **With Early Stopping:** May finish in 4-7 epochs (~1-1.5 hours)

**Training Configuration:**

```python
# Optimized for RTX 2000 Ada (8GB VRAM)
per_device_train_batch_size = 2
gradient_accumulation_steps = 8  # Effective batch size = 16
num_train_epochs = 10
bf16 = True  # Native for Ada GPUs (faster than fp16)
tf32 = True  # Hardware acceleration for matrix ops
dataloader_num_workers = 4  # Optimized for batch_size=2
learning_rate = 2e-4
warmup_steps = 50
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

- Increase `lora_dropout` from 0.1 to 0.15
- Reduce `num_train_epochs` from 10 to 5
- Check if you have enough samples per gloss (need 3-5 minimum)

---

### Data Pipeline Summary

```
Raw WLASL Dataset (2,000+ words)
         ↓
    Filter by 320-word glossary
         ↓
    Download videos from YouTube
         ↓
    Validate videos (remove corrupted)
         ↓
    [Optional] Preprocess to .pt tensors
         ↓
    Stratified train/test split (by gloss)
         ↓
    Training with dynamic frame sampling
```

### Current Functionality

#### Basic Video Processing

```python
from src.model.model_loader import VLMModelLoader

# Initialize the model loader
vlm_loader = VLMModelLoader()

# Process a video file
result = vlm_loader.generate_response("path/to/your/video.mp4")

# Clean up resources
vlm_loader.shutdown()
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

Modify `src/model/params/vlm.yml` to customize:

- Model selection
- Inference prompts
- Generation parameters

```yaml
model: OpenGVLab/InternVL3-2B-hf
prompt: You are a sign language interpreter. Given the video input, provide a concise and accurate text translation of the sign language being communicated.
```

### Fine-tuning Configuration

The fine-tuning process uses PEFT with LoRA and custom data collation:

```python
# PEFT/LoRA Configuration
peft_config = LoraConfig(
    r=8,                            # LoRA rank (increased from 2 for better capacity)
    lora_alpha=16,                  # LoRA scaling (typically 2x rank)
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./vlm_finetuned",
    per_device_train_batch_size=2,  # Optimized for RTX 2000 Ada (8GB VRAM)
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    gradient_accumulation_steps=16, # Effective batch size = 32
    fp16=(device_type == "cuda"),   # fp16 only for CUDA, float32 for DirectML
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",          # Evaluate every epoch
    gradient_checkpointing=True,    # Reduce memory usage
    remove_unused_columns=False,    # Keep all dataset fields
    report_to="tensorboard",        # Real-time monitoring
    logging_dir="./tb_logs",
    optim="adamw_torch",
)

# Custom data collator handles:
# - Video frame sampling (4 frames by default, configurable)
# - Processor formatting with <video> placeholder
# - Label masking (prompt=-100, answer=token_ids)
# - Dynamic batch filtering for corrupted videos
# - FP16 tensor conversion for memory efficiency
```

**Key Features:**

- **Gradient Checkpointing**: Reduces VRAM usage by ~40% (trades compute for memory)
- **4-bit Quantization**: Model loaded in 4-bit reduces memory by ~75%
- **LoRA (r=16)**: Only trains ~0.17% of parameters (~2M params) for efficiency
- **BF16 + TF32**: Native Ada GPU acceleration for 15-20% speedup over FP16
- **Optimized Workers**: dataloader_num_workers=4 prevents CPU bottleneck
- **Cosine Scheduler**: Learning rate decay for better convergence
- **Early Stopping**: Saves time by stopping when validation loss plateaus
- **Stratified Splitting**: Ensures each sign word appears proportionally in train/test
- **Dynamic Filtering**: Automatically skips corrupted videos during training
- **Chat Templates**: Proper instruction formatting with `<video>` token injection
- **TensorBoard**: Real-time loss curves and metrics visualization with custom callback

**Memory Breakdown (RTX 2000 Ada 8GB):**

```
Base Model (4-bit):     ~2GB
LoRA Adapters (r=16):   ~0.8GB
Activations (bs=2):     ~2GB
Grad Checkpointing:     Saves ~1.5GB
Optimizer States:       ~1.2GB
------------------------------
Total Usage:            ~6-7GB / 8GB (safe margin)
Peak During Eval:       ~7GB (acceptable)
```

**Training Speed Optimizations:**

| Optimization                    | Impact             | Notes                           |
| ------------------------------- | ------------------ | ------------------------------- |
| **BF16 instead of FP16**        | +15% speed         | Native on Ada GPUs              |
| **TF32 enabled**                | +10% speed         | Hardware matrix acceleration    |
| **dataloader_num_workers=4**    | +20% speed         | Prevents CPU bottleneck         |
| **gradient_accumulation=8**     | Better convergence | Updates weights more frequently |
| **Early stopping (patience=3)** | Saves 3-6 epochs   | Typical best model at epoch 5-7 |

**Expected Total Training Time:**

- **Without early stopping**: ~2-2.5 hours (10 epochs)
- **With early stopping**: ~1-1.5 hours (5-7 epochs typically)
- **Per epoch**: ~12-15 minutes (RTX 2000 Ada)

## 🔧 Hardware Requirements

### Minimum Requirements

- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **CPU**: Multi-core processor

### GPU Requirements (Optional but Recommended)

- **NVIDIA GPU**: GTX 1060 / RTX 2060 or better with 6GB+ VRAM
- **AMD GPU**: RX 6600 or better with DirectML support
- **VRAM**: 4GB minimum, 8GB+ recommended

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
│       ├── 12345.pt         # Tensor: (4, 3, H, W) - 4 frames, RGB, fp16
│       └── ...
├── raw_data/                # Original WLASL JSON files
├── filter_data.py           # Filters dataset by 320-word glossary
├── video_downloader.py      # Downloads videos from YouTube
└── validate_videos.py       # Validates videos and removes corrupted ones
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
# Test model loading and inference
cd src
python -m model.model_loader

# Test data processing pipeline
cd model/finetune/data_engineering
python filter_data.py

# Test dataset loader with stratified split
cd ../../../src
python -c "from model.finetune.dataset import DatasetLoader; dl = DatasetLoader(); dl._train_test_split()"

# Test fine-tuning pipeline (once preprocessing is complete)
cd src
python -m model.finetune.train
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

## 📊 Performance

### Current Benchmarks (Nov 25, 2025)

#### Base Model (Inference)

- **Model**: InternVL3.5-2B-hf (upgraded from InternVL3-2B)
- **Model Loading**: ~8-9 seconds on NVIDIA RTX 2000 Ada
- **Video Processing**: ~2-5 seconds per video (4 frames)
- **Memory Usage**: ~2-3GB VRAM (with 4-bit quantization)

#### Fine-tuning Process

- **Dataset Size**:

  - Total WLASL videos downloaded: ~1,892
  - After validation: ~1,603 videos (84.7% success rate)
  - Training split: 1,180 samples (73.6%)
  - Test split: 423 samples (26.4%)
  - Glossary: 320 core sign language words

- **Training Configuration**:

  - Hardware: NVIDIA RTX 2000 Ada (8GB VRAM)
  - Batch size: 2 (real) × 8 (grad_accum) = 16 (effective)
  - Epochs: 10 (with early stopping patience=3)
  - Learning rate: 2e-4 with cosine decay and 50 warmup steps
  - Precision: BF16 + TF32 (Ada GPUs) / FP32 (DirectML)
  - LoRA: r=16, alpha=32, dropout=0.1 (~2M trainable params)

- **Training Time Estimates**:

  - Per step: ~5-7 seconds (with optimizations)
  - Per epoch: ~148 steps × 6s = ~12-15 minutes
  - Full training (10 epochs): ~2-2.5 hours
  - With early stopping: ~1-1.5 hours (typically stops at epoch 5-7)

- **Memory Requirements**:

  - Peak VRAM usage: ~6-7GB / 8GB
  - Trainable parameters: 2,097,152 (~0.17% of 1.2B total)
  - Model size on disk: ~2GB (4-bit quantized)
  - Checkpoint size: ~8MB per epoch (LoRA adapters only)

- **Data Loading**:
  - Without preprocessing: ~1-2s per video (decord)
  - With preprocessing: ~0.5-1s per video (pre-sampled frames)
  - Speedup: ~40-60% faster with preprocessing

### Optimization Features

- **4-bit quantization** (CUDA only): Reduces memory usage by ~75%
- **BF16 precision** (Ada GPUs): 15% faster than FP16, better numerical stability
- **TF32 acceleration** (Ada GPUs): Hardware matrix multiplication speedup
- **DirectML support**: AMD GPU acceleration (float32 precision)
- **Gradient checkpointing**: Saves ~40% VRAM during training
- **Gradient accumulation**: Simulates large batch sizes without OOM
- **LoRA/PEFT (r=16)**: Trains only ~0.17% of parameters for efficiency
- **Cosine learning rate scheduler**: Better convergence than constant LR
- **Early stopping**: Automatically stops when model plateaus (saves time)
- **Optimized dataloader workers**: Prevents CPU bottleneck (workers=4 for batch_size=2)
- **Chat template integration**: Proper instruction formatting with `<video>` tokens
- **Dynamic video filtering**: Automatically skips corrupted videos during training
- **Skip-if-exists downloads**: Resume interrupted video downloads seamlessly
- **Efficient memory cleanup**: Aggressive garbage collection and cache clearing
- **Configurable generation parameters**: Control output length, sampling strategy
- **TensorBoard integration**: Real-time loss visualization with custom callback
- **Stratified dataset splitting**: Ensures balanced gloss representation in train/test
- **Optional preprocessing**: Pre-extract frames for 40-60% faster data loading

### Dataset Statistics & Health

- **Total WLASL Vocabulary**: ~2,000+ words
- **Filtered Glossary**: 320 essential words
- **Videos Downloaded**: ~1,892
- **Videos Validated**: ~1,603 (84.7% success rate)
- **Corrupted Videos**: ~289 (15.3% - automatically filtered)
- **Video Sources**: YouTube via WLASL dataset
- **Processing Pipeline**: Automated filtering, validation, and dataset generation
- **Train/Test Split**: 73.6% / 26.4% (stratified by gloss)
- **Frame Sampling**: 4 frames per video (configurable via NUM_FRAMES)

### Known Issues & Solutions

| Issue                               | Cause                          | Solution                                                         |
| ----------------------------------- | ------------------------------ | ---------------------------------------------------------------- |
| **moov atom not found**             | Corrupted video downloads      | Run `validate_videos.py` before training                         |
| **Gradient checkpointing warnings** | `use_cache=True` conflict      | Use `prepare_model_for_kbit_training()` (✅ FIXED)               |
| **OOM errors**                      | Batch size too large           | Reduce to batch_size=1, increase grad_accum                      |
| **Training stuck at 0% with skips** | Too many corrupted videos      | Validate dataset first with `validate_videos.py`                 |
| **Slow data loading**               | Decord processing overhead     | Use `preprocess_videos.py` for faster loading                    |
| **Loss not decreasing**             | Learning rate or config issue  | Check TensorBoard, try increasing LoRA r to 32                   |
| **Model predictions are bad**       | Insufficient training capacity | Increase LoRA r from 16 to 32, simplify prompt, check NUM_FRAMES |
| **Training too slow**               | Suboptimal configuration       | Use bf16=True, tf32=True, reduce workers to 4 (✅ OPTIMIZED)     |

## 🔮 Future Enhancements

### Short-term Goals (Next 2-4 weeks)

- **Model evaluation and inference pipeline** for trained checkpoints
- Automated metrics computation (accuracy, F1, per-gloss performance)
- Best checkpoint selection based on validation loss
- Hyperparameter optimization and experiment tracking
- Inference optimization (caching, batching)
- Model versioning and checkpoint management

### Medium-term Goals (1-3 months)

- REST API implementation with FastAPI
- WebSocket support for real-time video streaming
- Model serving infrastructure (TorchServe / Triton)
- Docker containerization for easy deployment
- Cloud deployment options (AWS/GCP/Azure)
- Mobile app integration (ONNX export)

### Long-term Vision (3-6 months)

- Support for multiple sign languages (ASL, BSL, etc.)
- Advanced fine-tuning strategies (QLoRA, full fine-tuning comparison)
- Continuous learning from user feedback
- Real-time video streaming optimization
- Custom model architectures for sign language
- Multi-modal input support (video + text context)
- Explainability features (attention visualization)

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
2. **Use Preprocessing**: Run `preprocess_videos.py` for 40-60% speedup
3. **Monitor GPU Usage**: Use `nvidia-smi` to check utilization
4. **Gradient Accumulation**: Use instead of increasing batch size if OOM
5. **FP16 Training**: Always enable on CUDA for 2x speedup
6. **Dataloader Workers**: Increase `dataloader_num_workers` if CPU bottleneck

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
