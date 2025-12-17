import torch
import yaml
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import BitsAndBytesConfig
import torch_directml
import time
import os
import torch.nn as nn
from peft import PeftModel, PeftConfig
from pathlib import Path

HF_TOKEN = os.getenv('HF_TOKEN')
NUM_FRAMES = 8 # Increased from 4 to 8
PREPROCESSED_DIR = "preprocessed_frames"


with open("../../params/vlm.yml", "r") as f:
    params = yaml.safe_load(f)

training_args = params.get("training_arguments", {})
lora_config = params.get("lora_config", {})


class SignLanguageClassifier(nn.Module):
    """
    Classification head on top of InternVL2 + LoRA.
    
    Architecture breakdown:
        1. InternVL2 base model (frozen)
        2. LoRA adapters (frozen) 
        3. Pooling layer (to convert sequence → single vector)
        4. Classification MLP (trainable)
    """
    def __init__(self, verbose, vlm_model, num_classes, hidden_dim, pooling="mean", dropout=0.1):
        super().__init__()

        self.vlm = vlm_model
        self.verbose = verbose
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim//2, self.num_classes)
        )

        self.__init_weights()
        if self.verbose:
            print(f"\n✅ Classification head created:\n")
            print(f"   Input dim: {hidden_dim}")
            print(f"   Hidden dim: {hidden_dim // 2}")
            print(f"   Output dim: {num_classes}")
            print(f"   Pooling: {pooling}")
            print(f"   Dropout: {dropout}")
        pass

    def __init_weights(self):
        """
        Initialize classifier weights using Xavier initialization.
        
        Why Xavier initialization?
        - Keeps signal variance consistent across layers
        - Prevents vanishing/exploding gradients
        - Standard practice for classification heads
        """

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        pass


    def pool_features(self, hidden_states):
        """
        Pool sequence of hidden states into single vector.
        
        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_dim)
                - batch_size: Number of videos in batch
                - seq_len: Number of tokens (varies based on video + prompt)
                - hidden_dim: Feature dimension (e.g., 4096)
        
        Returns:
            pooled: Tensor of shape (batch_size, hidden_dim)
        
        Example:
            Input: (4, 150, 4096)  # 4 videos, 150 tokens each, 4096 features
            Output: (4, 4096)       # 4 videos, 4096 features each
        """

        if self.pooling == "last":
            return hidden_states[:, -1, :]  # Take last token

        elif self.pooling == "mean":
            return hidden_states.mean(dim=1)  # Mean over seq_len
        
        elif self.pooling == "max":
            return hidden_states.max(dim=1)[0]
        
        elif self.pooling == "cls":
            return hidden_states[:, 0, :]  # Assume first token is [CLS]
        
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        pass

    def forward(self, pixel_values, input_ids, attention_mask=None):
        """
        Forward pass for classification.
        
        Args:
            pixel_values: Video frames
                Shape: (batch, num_frames, channels, height, width)
                Example: (4, 8, 3, 448, 448)
            input_ids: Tokenized text prompt
                Shape: (batch, seq_len)
                Example: (4, 20) for "What sign is being shown?"
            attention_mask: Mask for padding tokens (optional)
                Shape: (batch, seq_len)
        
        Returns:
            logits: Classification logits
                Shape: (batch, num_classes)
                Example: (4, 500) - raw scores for each class
        
        Training example:
            Input: Video of "book" sign
            Output logits: [0.1, 2.5, -0.3, ...] (500 values)
            After softmax: [0.02, 0.85, 0.001, ...] (85% confident it's "book")
        """

        outputs = self.vlm(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        last_hidden_state = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)

        pooled_output = self.pool_features(last_hidden_state)

        pooled_output = pooled_output.to(dtype=torch.float32)

        logits = self.classifier(pooled_output)  # (batch, num_classes)

        return logits

    @torch.no_grad()
    def predict(self, pixel_values, input_ids, attention_mask=None, top_k=5):
        """
        Make predictions with confidence scores.
        
        Args:
            pixel_values: Video frames
            input_ids: Tokenized prompt
            attention_mask: Attention mask
            top_k: Return top-k predictions
        
        Returns:
            Dictionary with:
                - 'logits': Raw logits (batch, num_classes)
                - 'probs': Probabilities after softmax (batch, num_classes)
                - 'top_k_indices': Top-k class indices (batch, top_k)
                - 'top_k_probs': Top-k probabilities (batch, top_k)
                - 'predicted_class': Most likely class index (batch,)
        
        Example output:
            {
                'logits': tensor([[2.1, -0.5, 3.2, ...]]),  # Raw scores
                'probs': tensor([[0.15, 0.01, 0.85, ...]]), # After softmax
                'top_k_indices': tensor([[2, 0, 15, 42, 8]]), # Top 5 classes
                'top_k_probs': tensor([[0.85, 0.15, 0.08, 0.05, 0.02]]),
                'predicted_class': tensor([2])  # Class 2 is most likely
            }
        """
        self.eval()  

        logits = self.forward(
            pixel_values=pixel_values,  
            input_ids=input_ids,
            attention_mask=attention_mask
            )
        

        probs = torch.softmax(logits, dim=-1)  # (batch, num_classes)

        top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)

        predicted_class = logits.argmax(dim=-1)  # (batch,)

        return {
            "logits": logits,
            "probs": probs,
            "top_k_indices": top_k_indices,
            "top_k_probs": top_k_probs,
            "predicted_class": predicted_class
        }
        pass

class FineTuneModelLoader:
    def __init__(self, verbose, lora_checkpoint_path):
        self.verbose = verbose
        self.lora_checkpoint_path = lora_checkpoint_path
        self.device = None
        self.device_type = None
        self.model_name = params["model"] # Set your model name here
        self.processor = None
        self.model = None
        self.num_classes = params["training_arguments"]["num_classes"]

        # Set device before loading model
        self.classifier_model = None
        self._set_device()
        
        # Load model
        self.load_model(self.lora_checkpoint_path)

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
    def load_model(self, lora_checkpoint_path: str):
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
            self.processor.video_processor.size = {"height": 448, "width": 448}

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

            
            if self.verbose:
                print("\n" + "="*70)
                print("🔧 Loading LoRA Adapter...")
                print("="*70)
                print(f"\n\n LoRA checkpoint: {lora_checkpoint_path}")


            
            checkpoint_path = Path(lora_checkpoint_path)
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"LoRA checkpoint not found at {lora_checkpoint_path}")
            
            lora_config = PeftConfig.from_pretrained(lora_checkpoint_path)

            if self.verbose:
                print(f"\nLoRA config loaded for : {lora_config.base_model_name_or_path}")
            
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_checkpoint_path,
                is_trainable=False 
            )
            

            if self.verbose:
                print(f"\n✅ LoRA Adapter loaded and merged successfully.")


            for param in self.model.parameters():
                param.requires_grad = False

            # Verify everything is frozen
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())

            if self.verbose:
                print(f"   Total VLM parameters: {total:,}")
                print(f"   Trainable parameters: {trainable:,}")
            
            if trainable == 0:
                print("   ✅ All VLM parameters are frozen")
            else:
                print("   ⚠️  WARNING: Some parameters are still trainable!")


            self.model.eval()
        
            print("\n" + "="*70)
            print("🔍 Inspecting Model to Determine Hidden Dimension...")

            hidden_dim = None
            config = self.model.config


            # Check various config locations for hidden_size
            if hasattr(config, "hidden_size"):
                hidden_dim = config.hidden_size
            elif hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
                hidden_dim = config.text_config.hidden_size
            elif hasattr(config, "llm_config") and hasattr(config.llm_config, "hidden_size"):
                hidden_dim = config.llm_config.hidden_size
            elif hasattr(config, "d_model"):
                hidden_dim = config.d_model

            if hidden_dim is None:
                print("⚠️  Could not auto-detect hidden dimension from config.")
                print(f"Config keys: {[k for k in config.__dict__.keys() if not k.startswith('_')]}")
                # Fallback for InternVL3-2B (likely 2048)
                hidden_dim = 2048 
                print(f"⚠️  Defaulting to {hidden_dim}. Check if this matches your model!")
            
            
            if self.verbose:
                print(f"   Detected hidden dimension: {hidden_dim}")
                print("\n" + "="*70)
                print("🔧 Creating Classification Head...")



            
            self.classifier_model = SignLanguageClassifier(
                vlm_model = self.model,
                verbose = self.verbose,
                hidden_dim=hidden_dim,
                num_classes= self.num_classes,
                # pooling=self.pooling,
                # dropout=dropout
            )

            if self.verbose:
                print("\n" + "="*70)
                print("📊 Model Meta Data")
                print("="*70)
                
                # Count classifier parameters
                classifier_params = sum(
                    p.numel() for p in self.classifier_model.classifier.parameters()
                )
                trainable_params = sum(
                    p.numel() for p in self.classifier_model.parameters() if p.requires_grad
                )
                total_params = sum(p.numel() for p in self.classifier_model.parameters())
                
                print(f"   VLM parameters (frozen): {total - classifier_params:,}")
                print(f"   Classifier parameters: {classifier_params:,}")
                print(f"   Total parameters: {total_params:,}")
                print(f"   Trainable parameters: {trainable_params:,}")
                print(f"   Trainable ratio: {trainable_params/total_params*100:.2f}%")
                
                # Memory estimate
                memory_mb = (trainable_params * 4) / (1024 ** 2)  # 4 bytes per float32
                print(f"   Estimated training memory: ~{memory_mb:.1f} MB")
                
                print("\n" + "="*70)
                print("✅ Model ready for training!")
                print("="*70 + "\n")

            end_time = time.time()
            elapsed_time = end_time - start_time

            print(f"\n\n✅ {self.model_name} VLM Model Loaded Successfully in ⏱️: {elapsed_time:.2f} seconds.")
            
            return self.classifier_model, self.processor


            

        except Exception as e:
            print(f"Error loading model: {e}")  

        pass
 
    pass



if __name__ == "__main__":
    loader = FineTuneModelLoader(verbose=True, lora_checkpoint_path="./vlm_finetuned_final")
    