import collections
import random
import dotenv
import torch
import os
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, Trainer, TrainingArguments, default_data_collator
from dataset import DatasetLoader
import cv2
# from torchvision.transforms import Compose, Lambda, Normalize, Resize, CenterCrop
# from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample
import decord
import yaml
import numpy as np
from collections import Counter



with open("../../params/vlm.yml", "r") as f:
    params = yaml.safe_load(f)


MODEL_CKPT = params["video_mae_params"]["pretrained_model_name"]
NUM_FRAMES = params["video_mae_params"]["num_frames"]
NUM_CLASSES = 282
REPEAT_FACTOR = params["video_mae_params"]["repeat_factor"]
# BATCH_SIZE = params["video_mae_params"]["per_device_train_batch_size"]


class VideoMAEDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, transform=None):

        self.loader = DatasetLoader(verbose=True)
        self.repeat_factor = REPEAT_FACTOR
        self.is_train=False


        self.train_list = self.loader.train_data
        self.val_list = self.loader.val_data
        self.test_list = self.loader.test_data

        # We need to know all possible glosses to assign IDs
        self.all_glosses = sorted(list(set(item['gloss'] for item in self.loader.dataset)))
        self.label2id = {label: i for i, label in enumerate(self.all_glosses)}
        self.id2label = {i: label for i, label in enumerate(self.all_glosses)}
        

        self.processor = VideoMAEImageProcessor.from_pretrained(MODEL_CKPT)
        
        if data_list == "train":
            self.is_train=True
            self.data_list = self.train_list
        
        elif data_list == "val":
            self.data_list = self.val_list

        elif data_list == "test":
            self.data_list = self.test_list

        
        training_data_list = []
        if data_list == "train":
            label_counts = {}
            for item in self.data_list:
                g = item["gloss"]
                label_counts[g] = label_counts.get(g, 0) + 1

            TARGET_PER_CLASS = 10 * self.repeat_factor  # Aim for 10 samples per class after expansion

            for item in self.data_list:
                gloss = item["gloss"]
                count = label_counts[gloss]

                r = max(1, int(TARGET_PER_CLASS / count))
                training_data_list.extend([item] * r)
            
            self.data_list = training_data_list
            random.shuffle(self.data_list)

        self.label2id = self.label2id
        self.transform = transform
        self.reader = decord.VideoReader
        pass

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item  = self.data_list[idx]

        video_path = "./../data_engineering/" + item["video_path"]

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        label_str = item["gloss"]

        try:
            vr = decord.VideoReader(video_path, num_threads = 1)
            total_frames = len(vr)


            if self.is_train:
                seg_size = float(total_frames - 1) / NUM_FRAMES
                indices = []
                for i in range(NUM_FRAMES):
                    start = int(np.round(seg_size * i))
                    end = int(np.round(seg_size * (i + 1)))
                    indices.append(random.randint(start, end))
                indices = np.array(indices)

            else:
                indices = np.linspace(0, len(vr)-1, NUM_FRAMES).astype(int)

            
            indices = np.clip(indices, 0, total_frames - 1)
            video = vr.get_batch(indices).asnumpy()  # (NUM_FRAMES, H, W, C)

            # ============================================================
            # 🅱️ SPATIAL AUGMENTATION (Flip & Crop)
            # ============================================================
            # 1. Random Horizontal Flip (50%)
            if random.random() < 0.5:
                video = np.flip(video, axis=2).copy() # Flip Width

            # 2. Random Brightness/Contrast (70%)
            if random.random() < 0.7:
                alpha = random.uniform(0.8, 1.2)  # Contrast
                beta = random.uniform(-20, 20)    # Brightness
                # Clip to valid range [0, 255] and cast back to uint8
                video = np.clip(alpha * video + beta, 0, 255).astype(np.uint8)

            # 3. Random Crop + Resize (80%)
            if random.random() < 0.8:
                h, w = video.shape[1:3] # (T, H, W, C)
                crop_scale = random.uniform(0.7, 1.0)  # Zoom 0-30%
                new_h, new_w = int(h * crop_scale), int(w * crop_scale)
                
                top = random.randint(0, h - new_h)
                left = random.randint(0, w - new_w)
                
                cropped = video[:, top:top+new_h, left:left+new_w, :]
                
                # Resize back to original size to keep tensor shape consistent
                # cv2.resize expects (width, height)
                resized = np.array([cv2.resize(frame, (w, h)) for frame in cropped])
                video = resized

            # 4. Random Rotation (30%)
            if random.random() < 0.3:
                angle = random.uniform(-10, 10) # +/- 10 degrees
                h, w = video.shape[1:3]
                # Calculate rotation matrix
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                # Apply to every frame
                video = np.array([cv2.warpAffine(frame, M, (w, h)) for frame in video])

            # 5. Random Temporal Speed (50%)
            # Note: Since we already sampled NUM_FRAMES earlier, this step effectively 
            # drops some frames and duplicates others to maintain the count, simulating speed changes.
            if random.random() < 0.5:
                speed = random.uniform(0.8, 1.2) # 80% to 120% speed
                n_frames = len(video)
                # Create new indices based on speed
                new_indices = np.linspace(0, n_frames-1, int(n_frames * speed))
                # Resample to get back to exactly NUM_FRAMES
                final_indices = np.linspace(0, len(new_indices)-1, NUM_FRAMES).astype(int)
                
                # Map back to the speed-adjusted indices
                selected_indices = new_indices[final_indices].astype(int)
                selected_indices = np.clip(selected_indices, 0, n_frames - 1)
                
                video = video[selected_indices]

            inputs = self.processor(list(video), return_tensors="pt")

            return {
                "pixel_values": inputs.pixel_values[0],
                "labels": torch.tensor(self.label2id[label_str], dtype=torch.long)
            }

        except Exception as e:
            raise RuntimeError(f"Error reading video {video_path}: {e}")
        pass

class VideoMAEFineTuner:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

        if self.verbose:
            print("🔄 Initializing VideoMAE Fine-Tuner...")
        self.image_processor = VideoMAEImageProcessor.from_pretrained(MODEL_CKPT)
        self.mean = self.image_processor.image_mean
        self.std = self.image_processor.image_std
        self.train_dataset = None
        self.val_dataset = None
        self.output_dir = params["video_mae_params"]["training_arguments"]["output_dir"]

        self.loader = DatasetLoader(verbose=True)

        # We need to know all possible glosses to assign IDs
        self.all_glosses = sorted(list(set(item['gloss'] for item in self.loader.dataset)))
        self.label2id = {label: i for i, label in enumerate(self.all_glosses)}
        self.id2label = {i: label for i, label in enumerate(self.all_glosses)}
        
        self.per_device_train_batch_size=params["video_mae_params"]["training_arguments"]["per_device_train_batch_size"]
        self.per_device_eval_batch_size=params["video_mae_params"]["training_arguments"]["per_device_eval_batch_size"]
        self.num_train_epochs=params["video_mae_params"]["training_arguments"]["num_train_epochs"]
        self.learning_rate=params["video_mae_params"]["training_arguments"]["learning_rate"]
        self.weight_decay=params["video_mae_params"]["training_arguments"]["weight_decay"]
        self.warmup_steps=params["video_mae_params"]["training_arguments"]["warmup_steps"]

        # self.val_transform = Compose([
        #     UniformTemporalSubsample(NUM_FRAMES),
        #     Lambda(lambda x: x / 255.0),
        #     Normalize(mean=self.mean, std=self.std),
        #     Resize((224, 224)),
        # ])

        self._load_data()
        self._load_model()
        pass


    def _load_data(self):
        self.train_dataset = VideoMAEDataset(data_list="train")
        self.val_dataset = VideoMAEDataset(data_list="val")

        # 👇 ADD THIS BLOCK
        print(f"\n📊 DATASET STATISTICS:")
        print(f"   - Training Samples (Expanded {self.train_dataset.repeat_factor}x): {len(self.train_dataset)}")
        print(f"   - Validation Samples: {len(self.val_dataset)}")
        print(f"   - Total Classes: {NUM_CLASSES}\n")

        # 🔍 VERIFICATION LOGIC START
        train_glosses = [item['gloss'] for item in self.train_dataset.data_list]
        counts = Counter(train_glosses)
        
        min_count = min(counts.values())
        max_count = max(counts.values())
        avg_count = sum(counts.values()) / len(counts)
        
        print(f"⚖️  BALANCE CHECK:")
        print(f"   - Min samples per class: {min_count}")
        print(f"   - Max samples per class: {max_count}")
        print(f"   - Avg samples per class: {avg_count:.1f}")
        pass

    def _load_model(self):
        self.model = VideoMAEForVideoClassification.from_pretrained(
            MODEL_CKPT,
            num_labels=NUM_CLASSES,
            label2id=self.label2id,
            id2label=self.id2label,
            ignore_mismatched_sizes=True,
        )

         # 1. Freeze the entire VideoMAE encoder first
        for param in self.model.videomae.parameters():
            param.requires_grad = False
            
        # 2. Unfreeze the last 4 layers of the encoder (VideoMAE Base has 12 layers)
        # This allows the model to learn high-level sign language features
        # while keeping low-level motion features stable.
        layers_to_unfreeze = 8
        encoder_layers = self.model.videomae.encoder.layer
        
        for i in range(len(encoder_layers) - layers_to_unfreeze, len(encoder_layers)):
            for param in encoder_layers[i].parameters():
                param.requires_grad = True

        # 3. Ensure LayerNorms are trainable (helps stability)
        for name, param in self.model.videomae.named_parameters():
            if "layernorm" in name:
                param.requires_grad = True
            
        print(f"❄️  FROZEN Early Layers. Unfrozen last {layers_to_unfreeze} layers + Classifier.")
        pass

    def get_glosses(self):
        self.all_glosses = sorted(list(set(item['gloss'] for item in self.loader.dataset)))
        print(self.all_glosses)

    
    @staticmethod
    def compute_metrics(eval_pred):
        """Static method to calculate accuracy during training"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return {"accuracy": (predictions == eval_pred.label_ids).mean()}


    def train(self):

        print("\n\n 🔄 Starting VideoMAE Fine-Tuning...")
        training_args = TrainingArguments(
            output_dir="./video_mae_finetuned",
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=self.num_train_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            logging_dir="./video_mae_finetuned_tb_logs",
            logging_steps=200,
            remove_unused_columns=False,
            dataloader_num_workers=5,
            metric_for_best_model="top5_accuracy",
            load_best_model_at_end=True,
            label_smoothing_factor=0.1, 
            fp16=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics = self.compute_metrics,
            data_collator = default_data_collator
        )

        trainer.train()
        print("💾 Saving Final Model...")
        trainer.save_model(self.output_dir)
        self.image_processor.save_pretrained(self.output_dir)
        pass




if __name__ == "__main__":
    finetuner = VideoMAEFineTuner()
    finetuner.train()
    # finetuner.get_glosses()