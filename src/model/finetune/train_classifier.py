import sklearn.metrics
import torch
from dataset import DatasetLoader
import yaml
import os
from pathlib import Path
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import decord
from model_loader import FineTuneModelLoader

HF_TOKEN = os.getenv('HF_TOKEN')
NUM_FRAMES = 8 # Increased from 4 to 8
PREPROCESSED_DIR = "preprocessed_frames"

with open("../params/vlm.yml", "r") as f:
    params = yaml.safe_load(f)

class ClassifierTrainer:
    def __init__(
       self,
       verbose,
    ):
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_loader = FineTuneModelLoader(verbose=True, lora_checkpoint_path="./vlm_finetuned/checkpoint-30")

        self.model = model_loader.classifier_model
        self.processor = model_loader.processor
        
        

        # load dataset
        dataset_loader = DatasetLoader(verbose=self.verbose)
        self.train_dataset = dataset_loader.train_data
        self.val_dataset = dataset_loader.val_data

        
        # We need to map string labels ("book") to integers (0)
        unique_glosses = sorted(set(item['gloss'] for item in self.train_dataset))
        gloss_to_id = {gloss: idx for idx, gloss in enumerate(unique_glosses)}
        num_classes = len(gloss_to_id)

        if self.verbose:
            print(f"Unique glosses: {unique_glosses}")
            print(f"Number of classes: {num_classes}")

        self.gloss_to_id = gloss_to_id
        self.id_to_gloss = {v: k for k, v in gloss_to_id.items()}


        self.num_frames = params["classifier_training_arguments"]["num_frames"]
        self.num_epochs = params["classifier_training_arguments"]["num_train_epochs"]
        self.batch_size = params["classifier_training_arguments"]["batch_size"]
        self.learning_rate = params["classifier_training_arguments"]["learning_rate"]
        self.warmup_steps = params["classifier_training_arguments"]["warmup_steps"]
        self.label_smoothing = params["classifier_training_arguments"]["label_smoothing_factor"]
        self.weight_decay = params["classifier_training_arguments"]["weight_decay"]
        self.save_dir = Path(params["classifier_training_arguments"]["save_dir"])
        self.save_dir.mkdir(exist_ok=True)
        
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)


        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5,
            T_mult=1,
            eta_min=1e-6
        )
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc=0.0
       
        pass

    def sample_frames(self,video_path):
        """
        Sample frames from a video file.
        """
        try:
            vr = decord.VideoReader(video_path)
            total = len(vr)
            # Uniformly sample indices
            indices = np.linspace(0, total - 1, self.num_frames).astype(int)
            frames = vr.get_batch(indices).asnumpy() # (T, H, W, C)
            # Convert to tensor and rearrange to (T, C, H, W) if needed by processor,
            # but usually processors handle (T, H, W, C) numpy arrays fine.
            # Let's keep it as numpy for the processor.
            return frames
        except Exception as e:
            print(f"Error reading video {video_path}: {e}")
            # Return black frames as fallback
            return np.zeros((self.num_frames, 448, 448, 3), dtype=np.uint8)
        pass

    def prepare_batch(self, batch_data):
        """
        Prepare a batch of data for the model.
        """

        videos=[]
        labels=[]
        texts = []


        for item in batch_data:
            #1. Load video frames
            video_path = item["video_path"]
            frames = self.sample_frames(video_path)  # (T, H, W, C)
            videos.append(frames)

            gloss = item["gloss"]
            if gloss in self.gloss_to_id:
                label_id = self.gloss_to_id[gloss]
                labels.append(label_id)

            else:
                continue


            texts.append("Describe the sign language gesture shown in the video.")

        if not videos:
            return None
        
        inputs = self.processor(
            text=texts,
            videos=videos,
            return_tensors="pt",
            padding=True
        )

        # 5. Convert labels to tensor
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return (
            inputs['pixel_values'].to(self.device),
            inputs['input_ids'].to(self.device),
            inputs.get('attention_mask', None).to(self.device) if 'attention_mask' in inputs else None,
            labels_tensor.to(self.device)
        )


    def train_epoch(self):

        self.model.train()
        total_loss = 0
        correct = 0
        total = 0


        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
            num_workers=0
        )

        progress_bar = tqdm(train_loader, desc="Training")

        for batch_data in progress_bar:
            batch_tuple = self.prepare_batch(batch_data)
            if batch_tuple is None:
                continue

            pixel_values, input_ids, attention_mask, labels = batch_tuple

            self.optimizer.zero_grad()

            logits = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss =- self.criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            

            self.optimizer.step()

            total_loss += loss.item() 
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct/total:.3f}"
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    
    @torch.no_grad()
    def validate(self):

        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        top5_correct = 0

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda x: x,
            num_workers=0
        )

        progress_bar = tqdm(val_loader, desc="Validation")

        for batch_data in progress_bar:
            batch_tuple = self.prepare_batch(batch_data)
            if batch_tuple is None:
                continue

            pixel_values, input_ids, attention_mask, labels = batch_tuple

            logits = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )      

            loss = self.criterion(logits, labels)
            total_loss += loss.item()

            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)


            top5_preds = logits.topk(5, dim=-1).indices

            for i in range(len(labels)):
                if labels[i] in top5_preds[i]:
                    top5_correct += 1

            total += labels.size(0)

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'top1': f"{correct/total:.3f}",
                'top5': f"{top5_correct/total:.3f}"
            })


        avg_loss = total_loss / len(val_loader)
        top1_acc = correct / total if total > 0 else 0
        top5_acc = top5_correct / total if total > 0 else 0
        
        return avg_loss, top1_acc, top5_acc

    def train(self):
        """
        Main training loop.
        """
        print(f"\n{'='*60}")
        print(f"🚀 Starting Training")
        print(f"{'='*60}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Num epochs: {self.num_epochs}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")


        for epoch in range(self.num_epochs):
            print(f"\n{'='*60}")
            print(f"🌟 Epoch {epoch+1}/{self.num_epochs}")
            print(f"{'='*60}\n")

            train_loss, train_acc = self.train_epoch()
            val_loss, val_top1, val_top5 = self.validate()

            # Step the scheduler
            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_top1)

            # 5. Print Summary
            print(f"\n📊 Epoch {epoch + 1} Summary:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Val Top-1 Acc: {val_top1:.4f}")
            print(f"   Val Top-5 Acc: {val_top5:.4f}")
            print(f"   LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # 6. Save Checkpoints
            # Save best model
            if val_top1 > self.best_val_acc:
                self.best_val_acc = val_top1
                self.save_checkpoint(epoch, is_best=True)
                print(f"   ✅ New best model saved! (Val Acc: {val_top1:.4f})")
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print(f"\n{'='*60}")
        print(f"✅ Training Complete!")
        print(f"Best Validation Accuracy: {self.best_val_acc:.4f}")
        print(f"{'='*60}\n")

        pass


    def save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint.
        """
        # We only save the classifier weights, not the whole VLM!
        # This saves massive amounts of space.
        checkpoint = {
            'epoch': epoch,
            'classifier_state_dict': self.model.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'gloss_to_id': self.gloss_to_id,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

        if is_best:
            path = self.save_dir / "best_classifier.pt"
        
        else:
            path = self.save_dir / f"classifier_epoch_{epoch+1}.pt"

        torch.save(checkpoint, path)
        print(f"   💾 Checkpoint saved at: {path}")
        pass


if __name__ == "__main__":
    Trainer = ClassifierTrainer(verbose=True)
    Trainer.train()
    pass