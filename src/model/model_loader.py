import gc
import os
import yaml
import time
from dotenv import load_dotenv
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
# Load environment variables from .env file
load_dotenv()
from pathlib import Path
import torch
import torch_directml
from peft import PeftModel
import decord

print(f"\n\n Warming Up File: Current Working Directory: {Path.cwd()}")

CONFIG = None
CHECKPOINT_NUM = "370"

# The 'with' statement is the best way to handle opening/closing files
with open('params/vlm.yml', 'r') as file:
    # Use yaml.safe_load() to parse the file
    # This converts the YAML into a Python dictionary (or list)
    try:
        CONFIG = yaml.safe_load(file)

    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")


# print(CONFIG["model"])
HF_TOKEN = os.getenv('HF_TOKEN')

class VLMModelLoader():
    def __init__(self):
        print("\n\n📍 Initializing VLMModelLoader...")
        self.model_name = CONFIG["model"]
        self.model=None
        self.processor=None
        self.prompt = CONFIG["prompt"]
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
        self.clear_memory()
        self.load_model()
        self.processor.video_processor.size={"height": 448, "width": 448}
        # self.lora_model_path = f"./finetune/vlm_finetuned_final"
        self.lora_model_path = f"./finetune/vlm_finetuned/checkpoint-{CHECKPOINT_NUM}"

    
    def load_model(self):
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

            if self.device_type == "dml":
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
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    quantization_config=bnb_config,
                    low_cpu_mem_usage=True,
                    token=HF_TOKEN,
                    device_map =self.device,
                    )
            end_time = time.time()
            elapsed_time = end_time - start_time


            # self.model.to(self.device)

            print(f"\n\n✅ {self.model_name} VLM Model Loaded Successfully in ⏱️: {elapsed_time:.2f} seconds.")

        except Exception as e:
            print(f"Error loading model: {e}")  

        pass

    
    def sample_frames(self, video_path: str, num_frames: int = 4):
        try:
            video_path = f"data_engineering/{video_path}"
            vr = decord.VideoReader(video_path)
            total = len(vr)
            indices = torch.linspace(0, total - 1, num_frames).long()
            frames = vr.get_batch(indices).asnumpy()   # (T, H, W, C)
            return frames
        except Exception as e:
            print(f"Error sampling frames from {video_path}: {e}")
            return None

    
    def generate_lora_response(self, video_path):
        lora_model_path = self.lora_model_path
        self.peft_model = PeftModel.from_pretrained(self.model, lora_model_path)
        self.peft_model.eval()

        video_tensor = self.sample_frames(video_path, num_frames=4)
        prompt = self.prompt

        conversation= [
            {"role": "user",
                "content": f"<video>\n{prompt}"}
        ]

        prompt_text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(text=[prompt_text], video=[video_tensor], return_tensors="pt", padding=True).to(self.device, dtype=torch.float16)

        # Generate
        with torch.no_grad():
            generated_ids = self.peft_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False, # Greedy decoding for reproducibility
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
            
        # Decode
        # The generated_ids includes the input_ids. We need to slice off the input part.
        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[0][input_len:]
        response = self.processor.decode(new_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def generate_lora_response_old(self, video_path):
        lora_model_path = self.lora_model_path
        self.peft_model = PeftModel.from_pretrained(self.model, lora_model_path)
        self.peft_model.eval()

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        print(f"\n\n🧠 Generating VLM Model Response for video: {video_path}...\n")
        video_path = Path(video_path)
        # print(f"Printing prompt: {self.prompt}")
        messages = [
            {
                "role": "user",
                "content": [ {
                        "type": "video",
                        "url": str(video_path)
                    },
                    {
                        "type":"text",
                        "text": self.prompt
                    }]
                    
                    
            }
        ]
        


        try:
            start_time = time.time()
            inputs = self.processor.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict =True,
                tokenize= True,
                add_generation_prompt = True,
                num_frames=4
            ).to(self.device, dtype=torch.float16)
            

            generation_config ={
                "max_new_tokens": 256,
                "repetition_penalty": 1.0,
                
            }

            output = self.peft_model.generate(**inputs, **generation_config)

            decoded_output = self.processor.decode(output[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

            result = decoded_output.strip()
            stop_time = time.time()
            elapsed_time = stop_time - start_time
            print(f"\n\n📝 Generated VLM Model Response: {result} in {elapsed_time} secs")

        except Exception as e:
            print(f"Error processing input: {e}")
        pass

    def generate_response(self, video_path):
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        print(f"\n\n🧠 Generating VLM Model Response for video: {video_path}...\n")
        video_path = Path(video_path)
        # print(f"Printing prompt: {self.prompt}")
        messages = [
            {
                "role": "user",
                "content": [ {
                        "type": "video",
                        "url": str(video_path)
                    },
                    {
                        "type":"text",
                        "text": self.prompt
                    }]
                    
                    
            }
        ]

        try:
            start_time = time.time()
            inputs = self.processor.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict =True,
                tokenize= True,
                add_generation_prompt = True,
                num_frames=4
            ).to(self.device, dtype=torch.float16)

            generation_config ={
                "max_new_tokens": 256,
                "repetition_penalty": 1.0,
            }

            output = self.model.generate(**inputs, **generation_config)

            decoded_output = self.processor.decode(output[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

            result = decoded_output.strip()
            stop_time = time.time()
            elapsed_time = stop_time - start_time
            print(f"\n\n📝 Generated VLM Model Response: {result} in {elapsed_time} secs")

        except Exception as e:
            print(f"Error processing input: {e}")
        pass

    def clear_memory(self):
        print("\n\n 🧹 Clearing Memory ... \n\n")
        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        print("\n\n 🧹 Memory Cleared. \n\n")

    def shutdown(self):
        print("\n\nShutting down and clearing memory...")
        del self.model
        del self.processor
        self.clear_memory()
        print("Shutdown complete.")
    pass




if __name__ == "__main__":
    vlm_loader = VLMModelLoader()
    print("\n\n Generating Response Test 1 Base MODEL")
    video_id= 69547
    video_path = f"/mnt/c/Users/aokeh/Documents/projects/personal/signLanguage/backend/sign-language-detector-backend/src/model/finetune/data_engineering/raw_videos/{video_id}.mp4"
    vlm_loader.generate_response(video_path)
    # print("\n\n Generating Response LORA MODEL NEW")
    # vlm_loader.generate_lora_response("/mnt/c/Users/aokeh/Documents/projects/personal/signLanguage/backend/sign-language-detector-backend/src/model/finetune/data_engineering/raw_videos/57096.mp4")
    print("\n\n Generating Response LORA MODEL OLD")
    vlm_loader.generate_lora_response_old(video_path)
    vlm_loader.shutdown()
    pass