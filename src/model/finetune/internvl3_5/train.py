# from model_loader import VLMModelLoader
from pathlib import Path
import time
from transformers import Trainer, TrainingArguments, AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, EarlyStoppingCallback
from datasets import Dataset
from dataset import DatasetLoader
import os
import json
import yaml
import torch
import torch_directml
from dotenv import load_dotenv
load_dotenv()
import decord
import gc
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torchvision import transforms
import random
import cv2
import numpy as np

# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

HF_TOKEN = os.getenv('HF_TOKEN')
NUM_FRAMES = 8 # Increased from 4 to 8
PREPROCESSED_DIR = "preprocessed_frames"

with open("../../params/vlm.yml", "r") as f:
    params = yaml.safe_load(f)

training_args = params.get("training_arguments", {})
lora_config = params.get("lora_config", {})

class VLMFineTuneTrainer():
    def __init__(self, verbose=True):
        """
        Initialize the VLM fine-tune trainer.

        Args:
            verbose (bool, optional): Enable verbose logging. Defaults to True.

        Attributes:
            model_name (str): HF model identifier from params
            model: Loaded base model
            processor: HF processor for model inputs
            device: Torch device in use (cuda/dml/cpu)
            peft_model: Model wrapped with LoRA adapters

        Returns:
            None
        """
        self.verbose = verbose
        self.per_device_train_batch_size = training_args["per_device_train_batch_size"]
        self.per_device_eval_batch_size = training_args["per_device_eval_batch_size"]
        self.gradient_accumulation_steps = training_args["gradient_accumulation_steps"]
        self.dataloader_num_workers = training_args["dataloader_num_workers"]
        self.learning_rate = training_args["learning_rate"]
        self.warmup_steps = training_args["warmup_steps"]
        self.label_smoothing_factor = training_args["label_smoothing_factor"]
        self.num_train_epochs = training_args["num_train_epochs"]
        self.lora_rank = lora_config["r"]
        self.lora_alpha = lora_config["lora_alpha"]
        self.dropout = lora_config["dropout"]

        # self.vlm_loader = VLMModelLoader()
        self.model_name = params["model"]
        self.model = None
        self.processor = None
        self.device = None
        self.device_type = None
        
        # Set device before loading model
        self._set_device()
        
        # Load model
        self.load_model()
       
        # CRITICAL: Prepare model for training (handles gradient checkpointing + use_cache)
        # This must be done BEFORE PEFT wrapping
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=True
        )

        if hasattr(self.model.language_model.config, 'use_cache'):
            if self.verbose:
                print("✅ Has USE CACHE", hasattr(self.model.language_model.config, 'use_cache'))        
            self.model.language_model.config.use_cache = False
            if self.verbose:
                print("✅ Disabled use_cache on language_model component")

        # FIX: Explicitly enable input gradients to suppress "None of the inputs have requires_grad=True" warning
        if hasattr(self.model, "enable_input_require_grads"):
            if self.verbose:
                print("✅ Enabling input gradients for the model to suppress warnings.")
            self.model.enable_input_require_grads()


        self.processor.video_processor.size={"height": 448, "width": 448}
        
        # Set LoRA config and wrap model
        self.peft_model = get_peft_model(self.model, self._set_lora_config())
        # self.peft_model.gradient_checkpointing_enable()
        # self.peft_model.config.use_cache = False  # Important for training

        # for name, param in self.peft_model.named_parameters():
        #     print(name, param.requires_grad)

        self.train()
        pass

    # ---------------------------------------------------------
    # DEVICE SETUP
    # ---------------------------------------------------------
    def _set_device(self):
        """
        Detect available hardware and set the trainer's device and device_type.

        Tries CUDA first, then DirectML, and finally falls back to CPU.

        Side effects:
            Updates self.device and self.device_type.
        """
        # --- NEW HARDWARE DETECTION LOGIC ---
        if torch.cuda.is_available():
            self.device_type = "cuda"
            self.device = torch.device("cuda")
            print(f"✅ Found NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        elif torch_directml.is_available():
            self.device_type = "dml"
            print(torch_directml.device())
            self.device = torch_directml.device() # Get the DML device
            
        else:
            self.device_type = "cpu"
            self.device = torch.device("cpu")
            print("❌ No GPU found. Falling back to CPU.")
    
    # ---------------------------------------------------------
    # MODEL LOADING
    # ---------------------------------------------------------
    def load_model(self):
        """
        Load the base Vision-Language model and prepare it for k-bit/LoRA training.

        This loads the processor and the model with appropriate quantization
        depending on the detected device. The loaded model is moved to
        `self.device`.

        Returns:
            None
        """
        if self.verbose:
            print(f"\n\n🚀 Starting {self.model_name} VLM Model Loading...")
        
        try:
            
            start_time = time.time()
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16"
            )

            self.processor = AutoProcessor.from_pretrained(self.model_name,trust_remote_code=True, token=HF_TOKEN)
             # 🚨 ADD THIS LINE HERE:
            self.processor.tokenizer.padding_side = "right"

            
            if self.device_type == "dml":
                if self.verbose:
                    print("--- Loading for AMD (DirectML) in float16 ---")
                    print("⚠️  4-bit quantization is NOT supported on DML")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,                    
                    dtype=torch.float16,    # Load in float16
                    token=HF_TOKEN,
                    low_cpu_mem_usage=True,
                
                ).to(self.device)
                
            else:
                if self.verbose:
                    print("--- Loading for CUDA in float16 ---")
                    print("✅  4-bit quantization is ENABLED for CUDA")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    quantization_config=bnb_config,
                    low_cpu_mem_usage=True,
                    token=HF_TOKEN,
                    # device_map =self.device,
                    )

                self.model = self.model.to(self.device)
            end_time = time.time()
            elapsed_time = end_time - start_time


            

            print(f"\n\n✅ {self.model_name} VLM Model Loaded Successfully in ⏱️: {elapsed_time:.2f} seconds.")

        except Exception as e:
            print(f"Error loading model: {e}")  

        pass

    # ---------------------------------------------------------
    # FRAME SAMPLING
    # ---------------------------------------------------------
    # def sample_frames(self, video_path: str, num_frames: int = NUM_FRAMES):
    #     """
    #     Sample `num_frames` evenly spaced frames from a video file.

    #     Args:
    #         video_path (str): Path to the video relative to `data_engineering/`.
    #         num_frames (int): Number of frames to sample. Defaults to NUM_FRAMES.

    #     Returns:
    #         numpy.ndarray|None: Array of frames with shape (T, H, W, C) or None on error.
    #     """
    #     try:
    #         video_path = f"data_engineering/{video_path}"
    #         vr = decord.VideoReader(video_path)
    #         total = len(vr)
    #         indices = torch.linspace(0, total - 1, num_frames).long()
    #         frames = vr.get_batch(indices).asnumpy()   # (T, H, W, C)
    #         return frames
    #     except Exception as e:
    #         print(f"Error sampling frames from {video_path}: {e}")
    #         return None

    # ---------------------------------------------------------
    # FRAME SAMPLING NEW
    # ---------------------------------------------------------
    def sample_frames(self, video_path: str, num_frames: int = NUM_FRAMES):
        """
        Sample `num_frames` evenly spaced frames from a video file.

        Args:
            video_path (str): Path to the video relative to `data_engineering/`.
            num_frames (int): Number of frames to sample. Defaults to NUM_FRAMES.

        Returns:
            numpy.ndarray|None: Array of frames with shape (T, H, W, C) or None on error.
        """
        try:
            # Use OpenCV instead of decord for better WSL stability
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                if self.verbose:
                    print(f"Error: Could not open video {video_path}")
                return None

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                cap.release()
                return None

            # Select indices evenly
            indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
            frames = []
            
            for idx in indices:
                # Seek to specific frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR (OpenCV default) to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    # If read fails, try to just append the last successful frame
                    if len(frames) > 0:
                        frames.append(frames[-1])
            
            cap.release()
            
            # Handle cases where video was too short or corrupt
            if len(frames) == 0:
                return None
            
            # Pad if we couldn't read enough frames (e.g. video shorter than num_frames)
            while len(frames) < num_frames:
                frames.append(frames[-1])
                
            return np.array(frames)
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return None

    # ---------------------------------------------------------
    # COLLATOR
    # ---------------------------------------------------------
    def video_collate(self, batch):
        """
        Collate function for DataLoader used during training/evaluation.

        - Loads preprocessed frames if available, otherwise samples frames on-the-fly.
        - Applies light augmentation (ColorJitter) with 50% probability.
        - Tokenizes prompts and prepares labels with masking so that only the
          assistant's generated tokens are used for loss computation.

        Args:
            batch (list): List of dataset items (dicts with keys: prompt, video_path, gloss)

        Returns:
            dict: Model-ready batch with input_ids, attention_mask, pixel_values, labels
        """
        # Sample frames and filter out any remaining corrupted videos
        # (should be rare if dataset was pre-validated)
        valid_items = []
        videos = []
        # is_training = self.peft_model.training  # Auto-detect mode
        # Define augmentations
        train_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            # transforms.RandomResizedCrop(size=(448, 448), scale=(0.9, 1.0), ratio=(0.9, 1.1)), # Optional: might be too aggressive
        ])

        for item in batch:
            # 1. Try to load preprocessed frames first

            video_path = item["video_path"][11:]

            pt_path = os.path.join(
                PREPROCESSED_DIR, video_path.replace(".mp4", ".pt")
            )

              
            frames = torch.load(pt_path, weights_only=True) if os.path.exists(pt_path) else None
            
            if frames is None:
                # Fallback to on-the-fly sampling
                if self.verbose:
                    print(f"⚠️ Preprocessed frames not found for {item['video_path']}, sampling on-the-fly.")
                
                frames = self.sample_frames(item["video_path"], num_frames=NUM_FRAMES)
                
                if frames is not None:
                    # sample_frames returns (T, H, W, C) numpy uint8
                    frames = torch.from_numpy(frames).float() / 255.0  # Keep (T, H, W, C)

            else:
                # Preprocessed frames are (T, H, W, C) in float16 (0-255 range)
                # Convert to float32 and normalize to 0-1 for transforms
                frames = frames.float() / 255.0
            

            if frames is not None:
                # Apply augmentation (ColorJitter expects (C, H, W) per frame, but we batch apply)
                # Temporarily permute to (T, C, H, W) for transforms, then back
                frames_tchw = frames.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
                
                if random.random() < 0.5:  # 50% chance
                    frames_tchw = train_transforms(frames_tchw)
                
                # Convert back to (T, H, W, C) for processor
                frames = frames_tchw.permute(0, 2, 3, 1)  # (T, C, H, W) -> (T, H, W, C)
                frames = (frames * 255.0).to(torch.uint8).numpy()
                
                videos.append(frames)
                valid_items.append(item)


            else:
                if self.verbose:
                    print(f"⚠️ Missing frames: {item['video_path']} - skipping sample.")
        
        # If all videos in batch are corrupted, this should be extremely rare
        # if you pre-validated the dataset
        if len(valid_items) == 0:
            if self.verbose:
                print("⚠️  Entire batch corrupted - this shouldn't happen with validated dataset!")
            # Return a minimal valid batch instead of raising exception
            # This prevents training from stopping
            return {
                "input_ids": torch.zeros((1, 1), dtype=torch.long),
                "attention_mask": torch.zeros((1, 1), dtype=torch.long),
                "labels": torch.full((1, 1), -100, dtype=torch.long),  # -100 = ignore in loss
                "pixel_values": torch.zeros((1, 1, 3, 224, 224), dtype=torch.float16),
            }
        
        prompts = [item["prompt"] for item in batch]
        answers = [item["gloss"] for item in batch]

        # ---------------------------------------------------------
        # 2. Apply Chat Template (The NEW part)
        # ---------------------------------------------------------
        
        # A. Create Full Conversation (User + Assistant)
        # We inject <video> manually into the content string
        full_conversations = [
            [
                {"role": "user", "content": f"<video>\n{p}"},
                {"role": "assistant", "content": a}
            ] for p, a in zip(prompts, answers)
        ]
        
        # tokenize=False gives us the raw string with special tokens (<s>, <|im_start|>, etc.)
        texts = [self.processor.apply_chat_template(c, tokenize=False) for c in full_conversations]
        
        # B. Create Prompt Only (User + Generation Trigger)
        # We need this to know where to stop masking
        prompts_only_conversations = [
            [{"role": "user", "content": f"<video>\n{p}"}] 
            for p in prompts
        ]
        
        # add_generation_prompt=True adds the "ASSISTANT:" token at the end
        prompts_only = [
            self.processor.apply_chat_template(c, tokenize=False, add_generation_prompt=True) 
            for c in prompts_only_conversations
        ]

        # 3. Tokenize everything (Main Call)
        model_inputs = self.processor(
            videos=videos,
            text=texts,
            return_tensors="pt", 
            padding=True
        )
        
        # 4. Tokenize prompts *without padding* to get their true lengths
        # -----------------------------------------------------------------
        # THIS IS THE ESSENTIAL FIX TO PREVENT THE CRASH
        prompt_token_lengths = [
            len(self.processor(
                text=[p],              # <--- Wrap in list
                videos=[v],            # <--- Wrap in list (remove v[None])
                return_tensors="pt"
            ).input_ids[0]) 
            for p, v in zip(prompts_only, videos)
        ]
        # -----------------------------------------------------------------

        # 5. Create and mask labels (rest of the code is fine)
        labels = model_inputs["input_ids"].clone()
        
        for i in range(len(labels)):
            prompt_len = prompt_token_lengths[i]
            labels[i, :prompt_len] = -100
            
            # Use the correct tokenizer attribute for pad_token_id
            pad_mask = model_inputs["input_ids"][i] == self.processor.tokenizer.pad_token_id
            labels[i, pad_mask] = -100

        model_inputs["labels"] = labels

 

        # ---- Cast video tensors to match model dtype ----
        for k in ["pixel_values", "video_pixel_values"]:
            if k in model_inputs:
                model_inputs[k] = model_inputs[k].to(dtype=torch.float16)

        # # =========================================================
        # # 🕵️ DEBUG: INSPECT LABELS (Add this block)
        # # =========================================================
        # if self.verbose and random.random() < 0.1: # Print for 10% of batches
        #     print(f"\n{'='*40}")
        #     print("🕵️  SANITY CHECK: DECODING BATCH")
        #     print(f"{'='*40}")
            
        #     # Grab the first sample in the batch
        #     debug_input_ids = model_inputs["input_ids"][0]
        #     debug_labels = model_inputs["labels"][0].clone()
            
        #     # 1. Decode the FULL Input (What the model sees)
        #     decoded_input = self.processor.tokenizer.decode(debug_input_ids, skip_special_tokens=False)
            
        #     # 2. Decode the LABEL (What the model learns)
        #     # Replace -100 with pad token so we can decode it
        #     debug_labels[debug_labels == -100] = self.processor.tokenizer.pad_token_id
        #     decoded_label = self.processor.tokenizer.decode(debug_labels, skip_special_tokens=False)
            
        #     print(f"👀 INPUT (Truncated):\n{decoded_input[:300]}...")
        #     print(f"\n🎯 LABEL (What is being learned):\n{decoded_label}")
        #     print(f"{'='*40}\n")
        # # =========================================================


        return model_inputs
    
    # ---------------------------------------------------------
    # LORA CONFIG
    # ---------------------------------------------------------
    def _set_lora_config(self):
        """
        Return the LoRA configuration used to wrap the base model.

        Returns:
            LoraConfig: Configuration object for PEFT/LoRA wrapping
        """
        return LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            lora_dropout=self.dropout,
            bias="none",
        )

    # ---------------------------------------------------------
    # TRAINING LOOP
    # ---------------------------------------------------------
    def train(self):
        """
        Orchestrate the fine-tuning process using Hugging Face `Trainer`.

        - Loads train/val splits via `DatasetLoader`.
        - Converts them to HF `Dataset` objects.
        - Configures `TrainingArguments` (including label smoothing).
        - Runs training, saves the final model to `./vlm_finetuned_final`.

        Returns:
            None
        """
        dataset_loader = DatasetLoader(verbose=self.verbose)
        train_dataset = dataset_loader.train_data
        val_dataset = dataset_loader.val_data # Use validation set for eval during training
        # test_dataset = dataset_loader.test_data # Keep test set for final evaluation

        # # Preprocess datasets
        # train_data = [self.preprocess_data(sample["video_path"]) for sample in train_dataset]
        # test_data = [self.preprocess_data(sample["video_path"]) for sample in test_dataset]

        
        hf_train_dataset = Dataset.from_list(train_dataset)
        hf_val_dataset = Dataset.from_list(val_dataset)

        training_args = TrainingArguments(
            output_dir="./vlm_finetuned",
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            num_train_epochs=self.num_train_epochs,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            dataloader_num_workers=self.dataloader_num_workers,
            dataloader_pin_memory=True,
            # fp16=True,
            bf16=True,
            tf32=True,
            learning_rate=self.learning_rate,                     # Added explicit learning rate
            warmup_steps=self.warmup_steps,                        # Added warmup
            lr_scheduler_type="cosine",             # Added cosine scheduler
            # weight_decay=0.01,                      # Added weight decay for regularization
            # bf16=False,
            logging_steps=30,
            save_strategy="epoch",
            eval_strategy="epoch",
            prediction_loss_only=True,              # <--- CRITICAL: Save RAM by not storing logits
            # eval_accumulation_steps=1,                 
            # save_total_limit=3,                     # Only keep best 3 checkpoints
            load_best_model_at_end=True,            # Load best model after training
            metric_for_best_model="eval_loss",      # Use eval loss to pick best
            gradient_checkpointing=True,
            remove_unused_columns=False,            # IMPORTANT
            report_to="tensorboard",
            logging_dir="./tb_logs",
            optim="adamw_torch",                    # ← SIMPLE ADAM
            label_names=["labels"],
            label_smoothing_factor=self.label_smoothing_factor,             # Added label smoothing
        )


        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=hf_train_dataset,
            eval_dataset=hf_val_dataset,
            data_collator=self.video_collate,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
        )

        start_time = time.time()
        if self.verbose:
            print("\n\n🧠 Starting Fine-Tuning...\n")

        trainer.train()

        trainer.save_model("./vlm_finetuned_final")

        end_time = time.time()
        elapsed_time = end_time - start_time

        if self.verbose:
            print(f"\n\n✅ Fine-Tuning Complete! in ⏱️: {elapsed_time:.2f} seconds.")

        
        pass

     # ---------------------------------------------------------
    
    
    # ---------------------------------------------------------
    # OLD METHODS FOR REFERENCES
    # ---------------------------------------------------------
    # COLLATOR (OLD - KEEP FOR REFERENCE)
    # def video_collate(self, batch):
    #     # Sample frames and filter out any remaining corrupted videos
    #     # (should be rare if dataset was pre-validated)
    #     valid_items = []
    #     videos = []

    #     for item in batch:
    #         pt_path = os.path.join(
    #             PREPROCESSED_DIR, item["video_path"].replace(".mp4", ".pt")
    #         )
    #         frames = torch.load(pt_path, weights_only=True) if os.path.exists(pt_path) else None

    #         if frames is not None:
    #             videos.append(frames)
    #             valid_items.append(item)
    #         else:
    #             if self.verbose:
    #                 print(f"⚠️ Missing frames: {item['video_path']} - skipping sample.")
        
    #     # If all videos in batch are corrupted, this should be extremely rare
    #     # if you pre-validated the dataset
    #     if len(valid_items) == 0:
    #         if self.verbose:
    #             print("⚠️  Entire batch corrupted - this shouldn't happen with validated dataset!")
    #         # Return a minimal valid batch instead of raising exception
    #         # This prevents training from stopping
    #         return {
    #             "input_ids": torch.zeros((1, 1), dtype=torch.long),
    #             "attention_mask": torch.zeros((1, 1), dtype=torch.long),
    #             "labels": torch.full((1, 1), -100, dtype=torch.long),  # -100 = ignore in loss
    #             "pixel_values": torch.zeros((1, 1, 3, 224, 224), dtype=torch.float16),
    #         }
        
    #     prompts = [item["prompt"] for item in valid_items]
    #     answers = [item["gloss"] for item in valid_items]

    #     # MUST match HF requirements: <video> placeholder!
    #     texts = [f"<video>USER: {p}\nASSISTANT: {a}" for p, a in zip(prompts, answers)]

    #     model_inputs = self.processor(
    #         videos=videos,
    #         text=texts,
    #         padding=True,
    #         return_tensors="pt",
    #     )

        
    #     # ---- Prepare labels ----
    #     model_inputs["labels"] = model_inputs["input_ids"].clone()


    #     # ---- Cast video tensors to match model dtype ----
    #     for k in ["pixel_values", "video_pixel_values"]:  # depends on what processor outputs
    #         if k in model_inputs:
    #             model_inputs[k] = model_inputs[k].to(dtype=torch.float16)


    #     return model_inputs

    # def train(self):
    #     if self.verbose:
    #         print("\n\n🧠 Starting Fine-Tuning with Custom Loop...\n")
    #     BATCH_SIZE = 1
    #     EPOCHS = 1
    #     GRAD_ACCUM_STEPS = 8
    #     LR = 1e-4
    #     FP16 = (self.device_type == "cuda")   # DirectML does NOT support autocast
    #     if self.verbose:
    #         print("\n🔥 Using custom PyTorch training loop (safe for DML)\n")


    #     if self.device_type == "dml":
    #         if self.verbose:
    #             print("⚠️  Enabling gradient checkpointing to reduce memory")
    #         # self.peft_model.gradient_checkpointing_enable()
       

    #     # Load dataset
    #     dataset_loader = DatasetLoader(verbose=self.verbose)
    #     train_dataset = dataset_loader.train_data
    #     test_dataset = dataset_loader.test_data

    #     # Torch dataloaders
    #     train_loader = DataLoader(
    #         train_dataset,
    #         batch_size=BATCH_SIZE,
    #         shuffle=True,
    #         collate_fn=self.video_collate
    #     )

    #     val_loader = DataLoader(
    #         test_dataset,
    #         batch_size=BATCH_SIZE,
    #         shuffle=False,
    #         collate_fn=self.video_collate
    #     )

    #     # Optimizer (simple Adam)
    #     optimizer = torch.optim.Adam(self.peft_model.parameters(), lr=LR)

    #     # AMP scaler (works only on CUDA)
    #     # scaler = torch.cuda.amp.GradScaler(enabled=FP16)

    #     # writer = SummaryWriter(log_dir="runs/vlm_finetune")
        
    #     global_step = 0

    #     self.peft_model.train()

    #     for epoch in range(EPOCHS):
    #         if self.verbose:
    #             print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

    #         for step, batch in enumerate(train_loader):

    #             # Move tensors to GPU/DML/CPU
    #             batch = {k: v.to(self.device) for k, v in batch.items()}

    #             # Forward pass
    #             with torch.cuda.amp.autocast(enabled=FP16):
    #                 outputs = self.peft_model(**batch)
    #                 loss = outputs.loss / GRAD_ACCUM_STEPS

    #             # Backward
    #             # scaler.scale(loss).backward()
    #             loss.backward()

    #             # Optimizer step
    #             if (step + 1) % GRAD_ACCUM_STEPS == 0:
    #                 # scaler.step(optimizer)
    #                 # scaler.update()
    #                 optimizer.step()
    #                 optimizer.zero_grad()

    #             if self.verbose:
    #                 print(f"  Step {step+1}, Loss = {loss.item():.4f}")
    #             # free intermediates
    #             del outputs, loss
    #             gc.collect()

    #             global_step += 1
    #             # writer.add_scalar("Train/Loss", loss.item(), global_step)

                

    #         # ----------------------------
    #         # Validation
    #         # ----------------------------
    #         self.peft_model.eval()
    #         total_val_loss = 0.0

    #         with torch.no_grad():
    #             for batch in val_loader:
    #                 batch = {k: v.to(self.device) for k, v in batch.items()}

    #                 outputs = self.peft_model(**batch)
    #                 total_val_loss += outputs.loss.item()

    #         total_val_loss /= len(val_loader)
    #         # writer.add_scalar("Validation/Loss", total_val_loss, epoch+1)

    #         if self.verbose:
    #             print(f"→ Validation Loss = {total_val_loss:.4f}")

    #         self.peft_model.train()

    #     # writer.close()
    #     print("\n🎉 Training completed!\n")




if __name__ == "__main__":
    vlm_finetune_trainer = VLMFineTuneTrainer()
