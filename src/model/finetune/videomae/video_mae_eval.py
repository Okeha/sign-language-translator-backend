import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor, default_data_collator
from sklearn.metrics import classification_report, accuracy_score
import yaml
import os
import json

# 1. Import the module so we can patch the global variable
import train_video_mae 
from train_video_mae import VideoMAEDataset

with open("../../params/vlm.yml", "r") as f:
    params = yaml.safe_load(f)


MODEL_CKPT = params["video_mae_params"]["pretrained_model_name"]

class VideoMAEEvaluator:
    def __init__(self, model_path="./video_mae_finetuned/checkpoint-77176", batch_size=4):
        """
        Args:
            model_path (str): Path to the saved model directory.
            batch_size (int): Batch size for inference.
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔄 Loading Model from {model_path}...")
        
        # 2. Load Model
        self.model = VideoMAEForVideoClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 3. 🛠️ FIX: Detect Model's expected frame count and patch the dataset
        model_num_frames = self.model.config.num_frames
        print(f"ℹ️ Model expects {model_num_frames} frames per video.")
        
        if train_video_mae.NUM_FRAMES != model_num_frames:
            print(f"⚠️ Mismatch detected! Dataset default is {train_video_mae.NUM_FRAMES}, patching to {model_num_frames}...")
            train_video_mae.NUM_FRAMES = model_num_frames

        # 4. Load Processor from the FINETUNED path (best practice)
        try:
            self.processor = VideoMAEImageProcessor.from_pretrained(model_path)
        except:
            print("⚠️ Could not load processor from checkpoint, falling back to base model.")
            self.processor = VideoMAEImageProcessor.from_pretrained(MODEL_CKPT)
        
        # Load Dataset (Test Split)
        print("🔄 Loading Test Dataset...")
        # Note: Using 'data_list_key' to match the class definition in train_video_mae.py
        # If your class uses 'data_list', change this argument name back.
        try:
            self.test_dataset = VideoMAEDataset(data_list_key="test")
        except TypeError:
             # Fallback if the class definition uses 'data_list'
            self.test_dataset = VideoMAEDataset(data_list="test")
        
        # Extract label mappings from the dataset
        self.id2label = self.test_dataset.id2label
        self.label2id = self.test_dataset.label2id
        self.classes = [self.id2label[i] for i in range(len(self.id2label))]
        
        print(f"✅ Loaded {len(self.test_dataset)} test samples.")

    def evaluate(self):
        print(f"🚀 Starting Evaluation on {self.device}...")
        
        dataloader = DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=default_data_collator,
            num_workers=2
        )

        all_preds = []
        all_labels = []
        all_probs = []

        # Inference Loop
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Get probabilities
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Store results
                all_probs.append(probs.cpu().numpy())
                all_preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # Concatenate all batches
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)

        # Extract video paths (assuming shuffle=False and dataset order is preserved)
        video_paths = ["./../data_engineering/" + item["video_path"] for item in self.test_dataset.data_list]

        self._compute_metrics(all_labels, all_preds, all_probs, video_paths)

    def _compute_metrics(self, y_true, y_pred, y_probs, video_paths):
        print("\n📊 --- Evaluation Results ---")
        
        # 1. Top-1 Accuracy
        top1_acc = accuracy_score(y_true, y_pred)
        print(f"🏆 Top-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.2f}%)")

        # 2. Top-5 Accuracy
        # Check if prediction is in top 5 probabilities
        top5_correct = 0
        detailed_results = []

        for i in range(len(y_true)):
            # Get indices of top 5 probabilities
            top5_indices = np.argsort(y_probs[i])[-5:][::-1] # Sort descending
            
            if y_true[i] in top5_indices:
                top5_correct += 1
            
            # Prepare JSON data
            true_gloss = self.id2label[y_true[i]]
            pred_gloss = self.id2label[y_pred[i]]
            top5_glosses = [self.id2label[idx] for idx in top5_indices]
            
            detailed_results.append({
                "video_path": video_paths[i],
                "ground_truth": true_gloss,
                "predicted_gloss": pred_gloss,
                "top_5_predictions": top5_glosses
            })
        
        top5_acc = top5_correct / len(y_true)
        print(f"🏅 Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.2f}%)")

        # Save detailed results to JSON
        output_json_path = "evaluation_results.json"
        with open(output_json_path, "w") as f:
            json.dump(detailed_results, f, indent=4)
        print(f"💾 Detailed results saved to {output_json_path}")

        # 3. Classification Report (Precision, Recall, F1 per class)
        print("\n📝 Generating Classification Report...")
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=self.classes, 
            zero_division=0
        )
        print(report)
        
        # Optional: Save report to file
        with open("evaluation_report.txt", "w") as f:
            f.write(f"Top-1 Accuracy: {top1_acc:.4f}\n")
            f.write(f"Top-5 Accuracy: {top5_acc:.4f}\n\n")
            f.write(report)
        print("💾 Report saved to evaluation_report.txt")

if __name__ == "__main__":
    # Ensure you point to the folder where trainer.save_model() saved the files
    # Usually "./video_mae_finetuned" or a specific checkpoint folder
    evaluator = VideoMAEEvaluator(model_path="./video_mae_finetuned/checkpoint-77176", batch_size=8)
    evaluator.evaluate()