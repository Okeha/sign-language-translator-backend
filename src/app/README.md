# Sign Language Recognition - Streamlit Application

Real-time sign language recognition using VideoMAE fine-tuned on WLASL dataset.

## Features

- 📹 Real-time camera capture
- 🤖 VideoMAE model inference (282 WLASL glosses)
- 📊 Confidence scores and top-5 predictions
- ⏱️ Inference latency tracking
- 🎨 Dark theme UI with custom styling

## Installation

### Using pip

```bash
pip install -r requirements.txt
```

### Using uv (recommended)

```bash
uv add streamlit opencv-python
```

## Usage

### For WSL2 Users (Camera Access Issue)

WSL2 doesn't have direct access to Windows cameras. Use one of these solutions:

**Option 1: Run from Windows (Recommended)**

```bash
# Double-click run_windows.bat in File Explorer
# Or run from Windows CMD:
src\app\run_windows.bat
```

This will start the server and open it in your Windows browser where the camera works!

**Option 2: Manual Windows Browser Access**

```bash
# In WSL, start with network access:
streamlit run src/app/streamlit_app.py --server.address=0.0.0.0

# Then open in Windows browser (not WSL):
# Find your WSL IP: wsl hostname -I
# Open: http://<WSL_IP>:8501
```

**Option 3: Upload Video Files**

- Use the "📁 Upload Video" option in the app
- Upload pre-recorded sign language videos (.mp4, .avi, .mov)

### For Native Linux/Windows Users

1. **Run the application:**

```bash
streamlit run src/app/streamlit_app.py
```

2. **Using the application:**
   - The app will open in your default browser (usually `http://localhost:8501`)
   - Choose input method:
     - **📷 Live Camera**: Record from webcam (requires camera permissions)
     - **📁 Upload Video**: Upload a pre-recorded video file
   - For live camera:
     - Grant camera permissions when prompted
     - Click **"Start Recording"** to capture 2 seconds of video
   - The model will process the frames and display:
     - Top prediction with confidence
     - Top-5 predictions
     - Inference latency
   - Click **"Make Prediction"** after recording to see results

## Configuration

### Model Path

The default model path is:

```python
model_path = "../model/finetune/videomae/video_mae_finetuned_final"
```

If your model is located elsewhere, update the path in the sidebar or modify `streamlit_app.py`.

### Frame Capture Settings

- **Buffer Size:** 32 frames captured
- **Sampled Frames:** 16 frames used for inference (uniformly sampled)
- **Recording Duration:** 2 seconds
- **FPS:** ~16 fps

You can adjust these in the code:

```python
FRAME_BUFFER_SIZE = 32  # Total frames to capture
NUM_FRAMES = 16         # Frames to sample for model
RECORDING_DURATION = 2  # Seconds to record
```

## Troubleshooting

### Camera not detected (WSL2)

**This is the most common issue!** WSL2 cannot access Windows USB devices (including cameras) directly.

**Solutions:**

1. **Use Windows browser** (easiest):
   - Run `run_windows.bat` from Windows
   - Or manually start with `--server.address=0.0.0.0` and access via WSL IP
2. **Upload video files instead**:
   - Switch to "📁 Upload Video" mode in the app
   - Use pre-recorded .mp4/.avi files
3. **Advanced: USB/IP forwarding**:
   - Follow [WSL USB guide](https://learn.microsoft.com/en-us/windows/wsl/connect-usb)
   - Requires usbipd-win on Windows

### Camera not detected (Native Linux/Windows)

- Ensure your webcam is connected and not being used by another application
- Check browser permissions for camera access
- Try changing the camera index in sidebar (0, 1, 2, etc.)
- Test camera with: `ls /dev/video*` (Linux) or Device Manager (Windows)

### "gio: Operation not supported" error

This is harmless - it's just WSL trying to open the browser. The server is still running.
Access the app through Windows browser instead.

### Model loading errors

- Verify the model path is correct
- Ensure the model files exist in the specified directory:
  - `config.json`
  - `model.safetensors` (or `pytorch_model.bin`)
  - `preprocessor_config.json`

### Performance issues

- **High latency:** Consider reducing frame buffer size or using a GPU
- **Low accuracy:** Ensure good lighting and clear hand gestures
- **Freezing:** Check if the model is too large for your system memory

### Import errors

If you see `ImportError` for `streamlit`, `cv2`, or `transformers`:

```bash
pip install -r src/app/requirements.txt
```

## Model Information

- **Base Model:** VideoMAE (MCG-NJU/videomae-base)
- **Dataset:** WLASL (282 classes)
- **Training:** Partial layer freezing (layers 8-11 + LayerNorms)
- **Input:** 16 frames, 224x224 resolution
- **Classes:** 282 American Sign Language glosses

## Upload Model to Hugging Face (Optional)

To share your model or load it from the cloud:

1. **Login to Hugging Face:**

```bash
huggingface-cli login
```

2. **Upload the model:**

```bash
huggingface-cli upload Okeha/sign-language-translator-videomae ./src/model/finetune/videomae/video_mae_finetuned_final .
```

3. **Update model path in app:**

```python
model_path = "Okeha/sign-language-translator-videomae"
```

## License

This project uses the VideoMAE model which is licensed under Apache 2.0.
