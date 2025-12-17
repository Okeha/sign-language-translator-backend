# Complete Guide: Building a Classification Head for VLM

## 📚 Table of Contents

1. [Conceptual Overview](#conceptual-overview)
2. [Architecture Deep Dive](#architecture-deep-dive)
3. [Step-by-Step Implementation](#step-by-step-implementation)
4. [Training Strategy](#training-strategy)
5. [Evaluation & Metrics](#evaluation--metrics)
6. [Common Issues & Solutions](#common-issues--solutions)

---

## 1. Conceptual Overview

### 🤔 What Problem Are We Solving?

**Your Current Approach (Generative):**

```python
# You ask the VLM: "What sign is this?"
input: <video> + "Identify the sign language gesture"
model.generate() → "the sign for book"
                 → "book is shown"
                 → "this represents book"

# Problem: Unreliable text that needs parsing!
```

**Classification Head Approach:**

```python
# You ask: "Which of these 100 signs is it?"
input: <video> + prompt
model.forward() → [0.001, 0.003, ..., 0.85, ..., 0.002]
                   ↑                    ↑
                  class_0           class_47 (book)

# Solution: Direct probability distribution!
```

### 🎯 Why Does This Work?

Your VLM has already learned to:

1. ✅ Extract visual features from sign language videos
2. ✅ Understand temporal dynamics (through LoRA fine-tuning)
3. ✅ Encode these into rich hidden representations

**The classification head simply learns:**

- "When I see hidden state pattern X, it means class Y"
- This is MUCH easier than generating coherent text!

---

## 2. Architecture Deep Dive

### 🏗️ Full Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT: Video (8 frames × 448×448×3)                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ PROCESSOR: Convert to tensors, normalize                     │
│   - Video → pixel_values_videos: [batch, frames, C, H, W]   │
│   - Text  → input_ids: [batch, seq_len]                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ VLM BACKBONE (FROZEN + LoRA weights)                         │
│                                                              │
│   Vision Encoder:                                            │
│   ┌──────────────────────────┐                              │
│   │ Video Frames             │                              │
│   │   ↓                      │                              │
│   │ Patch Embedding          │                              │
│   │   ↓                      │                              │
│   │ Temporal Attention       │  ← LoRA adapts this!        │
│   │   ↓                      │                              │
│   │ Spatial Attention        │  ← LoRA adapts this!        │
│   │   ↓                      │                              │
│   │ Visual Features          │                              │
│   │ [batch, num_patches, d]  │                              │
│   └──────────────────────────┘                              │
│                                                              │
│   Text Encoder + Fusion:                                     │
│   ┌──────────────────────────┐                              │
│   │ Text Tokens              │                              │
│   │   ↓                      │                              │
│   │ Text Embedding           │                              │
│   │   ↓                      │                              │
│   │ Cross-Attention          │  ← LoRA adapts this!        │
│   │ (Text ↔ Video)           │                              │
│   │   ↓                      │                              │
│   │ Transformer Layers       │  ← LoRA adapts this!        │
│   │   ↓                      │                              │
│   │ Hidden States            │                              │
│   │ [batch, seq_len, 4096]   │  ← This is what we extract!│
│   └──────────────────────────┘                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ POOLING: Reduce sequence to single vector                    │
│                                                              │
│ Option 1 - Last Hidden State:                               │
│   hidden_states[:, -1, :]  → [batch, 4096]                  │
│   ↑ Take the last token's representation                    │
│                                                              │
│ Option 2 - Mean Pooling:                                     │
│   hidden_states.mean(dim=1) → [batch, 4096]                 │
│   ↑ Average all token representations                       │
│                                                              │
│ Option 3 - CLS Token:                                        │
│   hidden_states[:, 0, :]  → [batch, 4096]                   │
│   ↑ Take the first (CLS) token                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ CLASSIFICATION HEAD (TRAINABLE!)                             │
│                                                              │
│   Input: [batch, 4096]                                       │
│      ↓                                                       │
│   Linear(4096 → 512)        ← Learnable weights W1          │
│      ↓                                                       │
│   LayerNorm(512)            ← Stabilize activations         │
│      ↓                                                       │
│   GELU()                    ← Non-linearity                 │
│      ↓                                                       │
│   Dropout(0.1)              ← Regularization                │
│      ↓                                                       │
│   Linear(512 → num_classes) ← Learnable weights W2          │
│      ↓                                                       │
│   Output: [batch, num_classes]  ← Raw logits!               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ LOSS & PREDICTION                                            │
│                                                              │
│ Training:                                                    │
│   loss = CrossEntropyLoss(logits, true_labels)              │
│   ↑ This automatically applies softmax internally           │
│                                                              │
│ Inference:                                                   │
│   probs = softmax(logits, dim=-1)                           │
│   top_k_probs, top_k_indices = torch.topk(probs, k=5)       │
│                                                              │
│   Example output:                                            │
│   top_k_indices = [47, 23, 8, 92, 15]                       │
│   top_k_probs   = [0.85, 0.08, 0.03, 0.02, 0.01]           │
│                     ↑ Predicted class!                       │
└─────────────────────────────────────────────────────────────┘
```

### 🔑 Key Insights

1. **Why Freeze the VLM?**
   - Your LoRA weights already understand sign language videos
   - Training the full VLM is slow and risks overfitting
   - The classification head is lightweight (~1M params vs 2B+ for VLM)
2. **Why Does the Classifier Work?**

   - VLM hidden states encode: "I see hand movement X in position Y"
   - Classifier learns: "This pattern = class 47 (book)"
   - Much simpler than: "Generate the word 'book' token by token"

3. **What About LoRA Weights?**

   ```python
   # LoRA adds small updates to VLM weights:
   W_new = W_original + α * (LoRA_A @ LoRA_B)

   # When you load LoRA adapter:
   model = PeftModel.from_pretrained(base_model, "checkpoint-60")
   # LoRA weights are merged into the forward pass automatically!
   ```

---

## 3. Step-by-Step Implementation

### Step 1: Load VLM with LoRA Adapter

```python
import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

# 1.1: Configure quantization (save memory)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Use 4-bit quantization
    bnb_4bit_use_double_quant=True,       # Double quantization for extra compression
    bnb_4bit_quant_type="nf4",           # NormalFloat4 (better than standard int4)
    bnb_4bit_compute_dtype=torch.float16 # Compute in FP16 for speed
)

# 1.2: Load base model
MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"  # or your model
base_model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    token=YOUR_HF_TOKEN,
)

# 1.3: Load LoRA adapter on top
vlm = PeftModel.from_pretrained(
    base_model,
    "./vlm_finetuned/checkpoint-60"  # Your trained LoRA weights
)

# 1.4: Load processor for inputs
processor = AutoProcessor.from_pretrained(MODEL_ID, token=YOUR_HF_TOKEN)
processor.video_processor.size = {"height": 448, "width": 448}
```

**What's happening here:**

- `base_model`: The original VLM (frozen weights)
- `PeftModel.from_pretrained()`: Loads your LoRA adapters
- The LoRA layers are automatically inserted into attention/MLP layers
- When you call `vlm.forward()`, it uses both base + LoRA weights

### Step 2: Extract Hidden States

```python
# 2.1: Prepare your input
video_tensor = load_your_video()  # Shape: [8, 3, 448, 448]
prompt = "Identify the sign language gesture shown in the video."

conversation = [{
    "role": "user",
    "content": f"<video>\n{prompt}"
}]

prompt_text = processor.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True
)

# 2.2: Process into model inputs
inputs = processor(
    text=[prompt_text],
    videos=[video_tensor],
    return_tensors="pt",
    padding=True
).to("cuda", dtype=torch.float16)

# inputs contains:
# - input_ids: [batch, seq_len] - tokenized text
# - attention_mask: [batch, seq_len] - which tokens to attend to
# - pixel_values_videos: [batch, frames, channels, height, width] - video frames

# 2.3: Get hidden states from VLM
with torch.no_grad():  # Don't compute gradients for VLM
    outputs = vlm(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        pixel_values_videos=inputs['pixel_values_videos'],
        output_hidden_states=True  # ← CRITICAL! This returns hidden states
    )

# outputs.hidden_states is a tuple of tensors, one per layer:
# (
#   layer_0: [batch, seq_len, hidden_dim],
#   layer_1: [batch, seq_len, hidden_dim],
#   ...
#   layer_N: [batch, seq_len, hidden_dim]  ← We want this!
# )

last_hidden_state = outputs.hidden_states[-1]  # [batch, seq_len, 4096]
```

**Key Points:**

- `output_hidden_states=True` makes the model return intermediate representations
- We take the **last layer** because it has the most processed features
- Shape is `[batch_size, sequence_length, hidden_dimension]`
- For Llama-3.2-11B-Vision, `hidden_dimension = 4096`

### Step 3: Pool Hidden States

```python
# 3.1: The problem - we have a sequence, but need one vector per video
# last_hidden_state.shape = [batch, seq_len, 4096]
# We need: [batch, 4096]

# Method 1: Last token (what the model uses for generation)
def pool_last_token(hidden_states, attention_mask=None):
    """Take the last token's hidden state."""
    # Simply take the last position in the sequence
    pooled = hidden_states[:, -1, :]  # [batch, 4096]
    return pooled

# Method 2: Mean pooling (more robust)
def pool_mean(hidden_states, attention_mask):
    """Average all tokens, ignoring padding."""
    # Expand attention mask to match hidden_states shape
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
    # [batch, seq_len, 1] → [batch, seq_len, 4096]

    # Multiply hidden states by mask (zeros out padding)
    masked_hidden = hidden_states * mask_expanded

    # Sum over sequence dimension
    sum_hidden = masked_hidden.sum(dim=1)  # [batch, 4096]

    # Count non-padding tokens
    sum_mask = mask_expanded.sum(dim=1)  # [batch, 4096]

    # Avoid division by zero
    sum_mask = torch.clamp(sum_mask, min=1e-9)

    # Average
    pooled = sum_hidden / sum_mask  # [batch, 4096]
    return pooled

# Method 3: CLS token (if your model uses one)
def pool_cls_token(hidden_states, attention_mask=None):
    """Take the first token (CLS token)."""
    pooled = hidden_states[:, 0, :]  # [batch, 4096]
    return pooled

# Use one of them:
pooled_features = pool_last_token(last_hidden_state, inputs['attention_mask'])
# pooled_features.shape = [batch, 4096]
```

**Why Pool?**

- VLM processes text as a **sequence**: `["<video>", "Identify", "the", "sign", ...]`
- Each token has its own hidden state
- For classification, we need **one vector per video**
- Pooling combines all token representations into one

**Which to Use?**

- **Last token**: Fast, works well if VLM is chat-tuned (expects response after last token)
- **Mean pooling**: More stable, uses all information
- **CLS token**: Traditional for BERT-style models
- **Try all three** and see which performs best!

### Step 4: Build Classification Head

```python
class ClassificationHead(nn.Module):
    """
    Maps VLM features to class logits.
    """
    def __init__(self, input_dim, num_classes, hidden_dim=512, dropout=0.1):
        """
        Args:
            input_dim: Size of VLM hidden states (e.g., 4096)
            num_classes: Number of sign language glosses to classify
            hidden_dim: Size of intermediate layer (None for direct projection)
            dropout: Dropout probability for regularization
        """
        super().__init__()

        if hidden_dim is not None:
            # Two-layer classifier with bottleneck
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),     # Project down
                nn.LayerNorm(hidden_dim),             # Normalize (helps training)
                nn.GELU(),                            # Non-linear activation
                nn.Dropout(dropout),                  # Prevent overfitting
                nn.Linear(hidden_dim, num_classes)    # Project to classes
            )
        else:
            # Single linear layer (simpler, less overfitting risk)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, num_classes)
            )

    def forward(self, pooled_features):
        """
        Args:
            pooled_features: [batch, input_dim] - VLM features
        Returns:
            logits: [batch, num_classes] - Raw scores (NOT probabilities!)
        """
        logits = self.classifier(pooled_features)
        return logits

# Create the head
num_classes = 100  # Number of sign language glosses in your dataset
classifier_head = ClassificationHead(
    input_dim=4096,      # VLM hidden size
    num_classes=num_classes,
    hidden_dim=512,      # Add intermediate layer
    dropout=0.1
)
```

**Why This Architecture?**

1. **Linear(4096 → 512)**: Compress VLM features

   - VLM features are high-dimensional and redundant
   - Compression acts as bottleneck regularization
   - Reduces parameters in final layer

2. **LayerNorm**: Normalize activations

   - Stabilizes training
   - Helps with gradient flow
   - Common in modern architectures

3. **GELU**: Smooth non-linearity

   - Better than ReLU for transformer-based models
   - Used in BERT, GPT, etc.

4. **Dropout**: Regularization

   - Randomly zeros out 10% of activations during training
   - Prevents overfitting to training set
   - Not used during inference

5. **Linear(512 → num_classes)**: Final projection
   - Maps to one score per class
   - These are **logits**, not probabilities

### Step 5: Combine into Full Model

```python
class VLMClassifier(nn.Module):
    """
    Complete model: VLM + Classification Head
    """
    def __init__(self, vlm, processor, num_classes, pooling='last', hidden_dim=512):
        super().__init__()
        self.vlm = vlm
        self.processor = processor
        self.pooling_strategy = pooling

        # Get VLM hidden dimension
        vlm_dim = vlm.config.text_config.hidden_size  # Usually 4096

        # Build classifier
        self.classifier = ClassificationHead(
            input_dim=vlm_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=0.1
        )

        # Freeze VLM (don't train it)
        for param in self.vlm.parameters():
            param.requires_grad = False

    def pool_features(self, hidden_states, attention_mask):
        """Pool hidden states based on strategy."""
        if self.pooling_strategy == 'last':
            return hidden_states[:, -1, :]
        elif self.pooling_strategy == 'mean':
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_hidden / sum_mask
        elif self.pooling_strategy == 'cls':
            return hidden_states[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling_strategy}")

    def forward(self, input_ids, attention_mask, pixel_values_videos):
        """
        Forward pass through VLM + classifier.

        Args:
            input_ids: [batch, seq_len] - Tokenized text
            attention_mask: [batch, seq_len] - Attention mask
            pixel_values_videos: [batch, frames, C, H, W] - Video frames

        Returns:
            logits: [batch, num_classes] - Class scores
        """
        # Get VLM features (frozen, no gradients)
        with torch.no_grad():
            vlm_outputs = self.vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos,
                output_hidden_states=True
            )

        # Extract and pool hidden states
        last_hidden_state = vlm_outputs.hidden_states[-1]
        pooled = self.pool_features(last_hidden_state, attention_mask)

        # Classify
        logits = self.classifier(pooled)

        return logits

    def predict(self, video_tensor, prompt, top_k=5):
        """
        Predict class for a single video.

        Args:
            video_tensor: [frames, C, H, W] - Video frames
            prompt: str - Text prompt
            top_k: int - Number of top predictions

        Returns:
            dict with predictions
        """
        self.eval()

        # Prepare input
        conversation = [{"role": "user", "content": f"<video>\n{prompt}"}]
        prompt_text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt_text],
            videos=[video_tensor],
            return_tensors="pt",
            padding=True
        ).to(next(self.parameters()).device, dtype=torch.float16)

        # Get predictions
        with torch.no_grad():
            logits = self.forward(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                pixel_values_videos=inputs['pixel_values_videos']
            )

            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)

            # Get top-k
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)

        return {
            'logits': logits[0].cpu(),
            'probabilities': probs[0].cpu(),
            'top_k_indices': top_indices[0].cpu().tolist(),
            'top_k_probabilities': top_probs[0].cpu().tolist(),
            'predicted_class': top_indices[0][0].item()
        }

# Create full model
model = VLMClassifier(
    vlm=vlm,
    processor=processor,
    num_classes=100,
    pooling='last',
    hidden_dim=512
)
```

---

## 4. Training Strategy

### Step 1: Prepare Dataset

```python
from torch.utils.data import Dataset, DataLoader

class SignLanguageDataset(Dataset):
    """
    Dataset that loads videos and returns processed tensors.
    """
    def __init__(self, data_samples, processor, num_frames=8):
        """
        Args:
            data_samples: List of dicts with keys:
                - 'video_path': path to video file
                - 'gloss': string label (e.g., "book")
                - 'label': integer class index (e.g., 47)
                - 'prompt': text prompt
            processor: VLM processor
            num_frames: number of frames to sample
        """
        self.data = data_samples
        self.processor = processor
        self.num_frames = num_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Load video frames
        video_tensor = self.load_video(sample['video_path'])

        # Prepare text
        conversation = [{
            "role": "user",
            "content": f"<video>\n{sample['prompt']}"
        }]
        prompt_text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        # Process
        inputs = self.processor(
            text=[prompt_text],
            videos=[video_tensor],
            return_tensors="pt",
            padding=True
        )

        # Remove batch dimension (DataLoader will add it back)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        return {
            **inputs,
            'label': torch.tensor(sample['label'], dtype=torch.long)
        }

    def load_video(self, video_path):
        """Load and sample frames from video."""
        import decord
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)

        # Sample evenly spaced frames
        indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
        frames = vr.get_batch(indices).asnumpy()  # [T, H, W, C]

        # Convert to tensor and rearrange
        frames = torch.tensor(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
        return frames

# Custom collate function for batching
def collate_fn(batch):
    """Stack batch items properly."""
    labels = torch.stack([item['label'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    pixel_values_videos = torch.stack([item['pixel_values_videos'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values_videos': pixel_values_videos,
        'labels': labels
    }

# Create dataloaders
train_dataset = SignLanguageDataset(train_data, processor)
val_dataset = SignLanguageDataset(val_data, processor)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,        # Adjust for your GPU
    shuffle=True,        # Randomize training order
    num_workers=0,       # 0 for Windows, 4+ for Linux
    collate_fn=collate_fn,
    pin_memory=True      # Faster GPU transfer
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,       # Don't shuffle validation
    num_workers=0,
    collate_fn=collate_fn,
    pin_memory=True
)
```

### Step 2: Setup Training

```python
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer - ONLY train classifier parameters!
trainable_params = [p for p in model.classifier.parameters()]
optimizer = AdamW(
    trainable_params,
    lr=1e-3,              # Higher LR since we're only training small head
    weight_decay=0.01,    # L2 regularization
    betas=(0.9, 0.999)    # Adam momentum parameters
)

# Learning rate scheduler
num_epochs = 20
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=num_epochs * len(train_loader),  # Total steps
    eta_min=1e-5          # Minimum LR
)

# Move model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

**Why These Hyperparameters?**

1. **Learning Rate (1e-3)**:

   - Higher than typical fine-tuning (1e-5)
   - We're training from scratch (random init)
   - Only training small head, not full model

2. **Weight Decay (0.01)**:

   - L2 regularization prevents overfitting
   - Pushes weights toward zero

3. **AdamW**:

   - Adaptive learning rate per parameter
   - Better weight decay than standard Adam

4. **Cosine Annealing**:
   - LR starts high, gradually decreases
   - Helps converge smoothly

### Step 3: Training Loop

```python
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()  # Enable dropout, etc.

    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values_videos = batch['pixel_values_videos'].to(device, dtype=torch.float16)
        labels = batch['labels'].to(device)

        # Forward pass
        optimizer.zero_grad()  # Reset gradients
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos
        )

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()  # Compute gradients

        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()
        scheduler.step()  # Update learning rate

        # Track metrics
        total_loss += loss.item()
        predictions = logits.argmax(dim=-1)  # Get predicted class
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        progress_bar.set_postfix({
            'loss': total_loss / (progress_bar.n + 1),
            'acc': 100 * correct / total,
            'lr': optimizer.param_groups[0]['lr']
        })

    avg_loss = total_loss / len(dataloader)
    avg_acc = 100 * correct / total

    return avg_loss, avg_acc

@torch.no_grad()  # Don't compute gradients during validation
def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()  # Disable dropout, etc.

    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Validation")

    for batch in progress_bar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values_videos = batch['pixel_values_videos'].to(device, dtype=torch.float16)
        labels = batch['labels'].to(device)

        # Forward pass
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos
        )

        # Compute loss
        loss = criterion(logits, labels)

        # Track metrics
        total_loss += loss.item()
        predictions = logits.argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        progress_bar.set_postfix({
            'loss': total_loss / (progress_bar.n + 1),
            'acc': 100 * correct / total
        })

    avg_loss = total_loss / len(dataloader)
    avg_acc = 100 * correct / total

    return avg_loss, avg_acc

# Full training loop
best_val_acc = 0.0

for epoch in range(num_epochs):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"{'='*60}")

    # Train
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, scheduler, device
    )

    # Validate
    val_loss, val_acc = validate(
        model, val_loader, criterion, device
    )

    # Print summary
    print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, 'best_classifier.pt')
        print(f"💾 Saved best model! (Val Acc: {val_acc:.2f}%)")

print(f"\n✅ Training complete! Best validation accuracy: {best_val_acc:.2f}%")
```

**What Happens During Training?**

1. **Forward Pass**:

   - Video + text → VLM (frozen) → hidden states
   - Hidden states → pooling → pooled features
   - Pooled features → classifier → logits

2. **Loss Computation**:

   - `CrossEntropyLoss(logits, labels)` computes:
     ```python
     # Internally:
     log_probs = log_softmax(logits)
     loss = -log_probs[batch_idx, true_label_idx].mean()
     ```
   - High loss = wrong predictions
   - Low loss = correct predictions

3. **Backward Pass**:

   - `loss.backward()` computes gradients:
     ```python
     # For each weight w in classifier:
     gradient = ∂loss/∂w
     ```
   - Gradients show which direction to adjust weights

4. **Optimizer Step**:

   - `optimizer.step()` updates weights:
     ```python
     # Simplified AdamW update:
     w_new = w_old - lr * gradient - weight_decay * w_old
     ```

5. **Repeat** until convergence!

---

## 5. Evaluation & Metrics

### Compute Top-1 and Top-5 Accuracy

```python
@torch.no_grad()
def evaluate_with_topk(model, test_loader, device, k=5):
    """
    Evaluate with Top-1 and Top-K accuracy.

    Returns:
        dict with metrics and detailed results
    """
    model.eval()

    all_predictions = []
    all_labels = []
    top_k_correct = 0
    total = 0

    for batch in tqdm(test_loader, desc="Evaluating"):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values_videos = batch['pixel_values_videos'].to(device, dtype=torch.float16)
        labels = batch['labels'].to(device)

        # Forward pass
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos
        )

        # Get predictions
        probs = torch.softmax(logits, dim=-1)

        # Top-1
        top1_pred = probs.argmax(dim=-1)
        all_predictions.extend(top1_pred.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        # Top-K
        _, topk_indices = torch.topk(probs, k=k, dim=-1)  # [batch, k]

        # Check if true label is in top-k
        for i in range(labels.size(0)):
            if labels[i] in topk_indices[i]:
                top_k_correct += 1
            total += 1

    # Calculate metrics
    from sklearn.metrics import accuracy_score, classification_report

    top1_acc = accuracy_score(all_labels, all_predictions)
    topk_acc = top_k_correct / total

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Top-1 Accuracy: {top1_acc*100:.2f}%")
    print(f"Top-{k} Accuracy: {topk_acc*100:.2f}%")
    print(f"\n{classification_report(all_labels, all_predictions)}")

    return {
        'top1_accuracy': top1_acc,
        f'top{k}_accuracy': topk_acc,
        'predictions': all_predictions,
        'labels': all_labels
    }

# Run evaluation
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
results = evaluate_with_topk(model, test_loader, device, k=5)
```

### Understanding the Metrics

**Top-1 Accuracy:**

```python
# For each sample:
predicted_class = argmax(probabilities)
correct = (predicted_class == true_class)

# Accuracy = % of samples where top prediction is correct
top1_acc = sum(correct) / total_samples
```

**Top-5 Accuracy:**

```python
# For each sample:
top_5_predictions = top_k(probabilities, k=5)  # [class_23, class_5, class_89, class_12, class_47]
correct = (true_class in top_5_predictions)

# Accuracy = % of samples where true class is in top 5
top5_acc = sum(correct) / total_samples
```

**Why Top-5 Matters:**

- Some signs are visually similar (e.g., "book" vs "read")
- Top-5 shows if the model is "close" even when wrong
- Useful for user interfaces (show top 5 suggestions)

---

## 6. Common Issues & Solutions

### Issue 1: Out of Memory (OOM)

**Symptoms:**

```
RuntimeError: CUDA out of memory
```

**Solutions:**

```python
# Solution 1: Reduce batch size
BATCH_SIZE = 2  # or even 1

# Solution 2: Reduce number of frames
NUM_FRAMES = 4  # instead of 8

# Solution 3: Use gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(...) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Solution 4: Clear cache between batches
import gc
torch.cuda.empty_cache()
gc.collect()
```

### Issue 2: Model Not Learning (Loss Stuck)

**Symptoms:**

```
Epoch 1: loss=4.605, acc=1%
Epoch 5: loss=4.602, acc=1%
Epoch 10: loss=4.601, acc=1%
```

**Diagnosis:**

```python
# Check if gradients are flowing
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: grad_norm={param.grad.norm().item()}")

# If all zeros → problem!
```

**Solutions:**

```python
# Solution 1: Check VLM is actually frozen
for param in model.vlm.parameters():
    assert not param.requires_grad, "VLM should be frozen!"

# Solution 2: Increase learning rate
optimizer = AdamW(params, lr=1e-2)  # Try 10x higher

# Solution 3: Simplify classifier
# Remove hidden layer, use single Linear
classifier = nn.Linear(4096, num_classes)

# Solution 4: Check labels are correct
print(f"Labels range: {min(all_labels)} to {max(all_labels)}")
print(f"Num classes: {num_classes}")
assert max(all_labels) < num_classes, "Label out of range!"
```

### Issue 3: Overfitting (Train Acc High, Val Acc Low)

**Symptoms:**

```
Epoch 10:
  Train Acc: 95%
  Val Acc:   25%  ← Big gap!
```

**Solutions:**

```python
# Solution 1: Increase dropout
classifier = ClassificationHead(dropout=0.3)  # instead of 0.1

# Solution 2: Add weight decay
optimizer = AdamW(params, weight_decay=0.1)  # instead of 0.01

# Solution 3: Use simpler model
classifier = nn.Linear(4096, num_classes)  # No hidden layer

# Solution 4: Data augmentation
# (Temporal jittering, cropping, etc.)

# Solution 5: Early stopping
if val_acc < best_val_acc_in_last_5_epochs:
    print("Stopping early - overfitting detected")
    break
```

### Issue 4: LoRA Weights Not Loading

**Symptoms:**

```
⚠️ WARNING: No LoRA modules found!
```

**Diagnosis:**

```python
# Check adapter files exist
import os
adapter_path = "./vlm_finetuned/checkpoint-60"
assert os.path.exists(f"{adapter_path}/adapter_config.json")
assert os.path.exists(f"{adapter_path}/adapter_model.bin") or \
       os.path.exists(f"{adapter_path}/adapter_model.safetensors")

# Check LoRA modules loaded
lora_modules = [n for n, m in model.vlm.named_modules() if 'lora' in n.lower()]
print(f"Found {len(lora_modules)} LoRA modules:")
print(lora_modules[:5])  # Should show names like 'base_model.model.layers.0.self_attn.q_proj.lora_A'
```

**Solutions:**

```python
# Solution 1: Check adapter path is correct
print(os.listdir("./vlm_finetuned/"))  # Should show checkpoint-60

# Solution 2: Load adapter after base model
base_model = AutoModelForImageTextToText.from_pretrained(...)
vlm = PeftModel.from_pretrained(base_model, adapter_path)  # Correct order!

# Solution 3: Verify PEFT version
# pip install peft --upgrade
```

### Issue 5: Slow Training

**Solutions:**

```python
# Solution 1: Use mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():  # Use FP16 for forward pass
        logits = model(...)
        loss = criterion(logits, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

# Solution 2: Increase num_workers
train_loader = DataLoader(..., num_workers=4)  # Parallel data loading

# Solution 3: Use pin_memory
train_loader = DataLoader(..., pin_memory=True)  # Faster GPU transfer

# Solution 4: Compile model (PyTorch 2.0+)
model = torch.compile(model)
```

---

## 7. Putting It All Together

Here's a complete minimal example:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

# 1. Load VLM with LoRA
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
base_model = AutoModelForImageTextToText.from_pretrained("your-model-id", quantization_config=bnb_config)
vlm = PeftModel.from_pretrained(base_model, "./vlm_finetuned/checkpoint-60")
processor = AutoProcessor.from_pretrained("your-model-id")

# 2. Build classifier
class SimpleClassifier(nn.Module):
    def __init__(self, vlm, num_classes):
        super().__init__()
        self.vlm = vlm
        self.classifier = nn.Linear(4096, num_classes)
        for param in self.vlm.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, pixel_values_videos):
        with torch.no_grad():
            outputs = self.vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos,
                output_hidden_states=True
            )
        pooled = outputs.hidden_states[-1][:, -1, :]
        logits = self.classifier(pooled)
        return logits

model = SimpleClassifier(vlm, num_classes=100).cuda()

# 3. Setup training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)

# 4. Train
for epoch in range(10):
    model.train()
    for batch in tqdm(train_loader):
        logits = model(
            batch['input_ids'].cuda(),
            batch['attention_mask'].cuda(),
            batch['pixel_values_videos'].cuda()
        )
        loss = criterion(logits, batch['labels'].cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            logits = model(
                batch['input_ids'].cuda(),
                batch['attention_mask'].cuda(),
                batch['pixel_values_videos'].cuda()
            )
            preds = logits.argmax(dim=-1)
            correct += (preds == batch['labels'].cuda()).sum().item()
            total += batch['labels'].size(0)

    print(f"Epoch {epoch+1}: Val Acc = {100*correct/total:.2f}%")

print("Done!")
```

---

## Summary

**The Key Idea:**

1. Your LoRA VLM already understands sign language videos → rich hidden representations
2. Classification head learns: "This hidden pattern = this class" (simple mapping!)
3. Much easier than generating text token-by-token

**Why It Works:**

- VLM does the hard work (video understanding) - already trained via LoRA
- Classifier does easy work (pattern matching) - trains in minutes
- Clean separation of concerns

**Expected Results:**

- **Without LoRA**: 20-30% accuracy (base VLM not adapted to sign language)
- **With LoRA generation**: 10-20% accuracy (generation is unreliable)
- **With LoRA + classifier**: 40-70% accuracy (best of both worlds!)

**Next Steps:**

1. Start with the minimal example above
2. Verify LoRA is loading correctly
3. Train for 10-20 epochs
4. Evaluate Top-1 and Top-5 accuracy
5. Iterate on architecture/hyperparameters

Good luck! 🚀
