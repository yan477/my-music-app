from flask import Flask, request, render_template, send_from_directory
import os
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import cv2
import numpy as np
import librosa
import music21
import threading
import uuid

# dictionary to track job statuses
jobs = {}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load AI model (optional, will load on first use)
pipe = None
controlnet = None

def load_model():
    global pipe, controlnet
    if pipe is None:
        print("Loading AI model...")
        try:
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
            )
            from diffusers import UniPCMultistepScheduler
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.enable_model_cpu_offload()
            print("Model loaded!")
        except Exception as e:
            print(f"Model loading failed: {e}")
            pipe = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('index.html')
    # handle POST: create a background job
    if 'source' not in request.files or 'reference' not in request.files:
        return 'Missing files', 400
    source_file = request.files['source']
    reference_file = request.files['reference']
    if source_file.filename == '' or reference_file.filename == '':
        return 'No selected file', 400
    source_path = os.path.join(app.config['UPLOAD_FOLDER'], f'source_{uuid.uuid4().hex}.jpg')
    reference_path = os.path.join(app.config['UPLOAD_FOLDER'], f'reference_{uuid.uuid4().hex}.jpg')
    source_file.save(source_path)
    reference_file.save(reference_path)
    # create job
    job_id = uuid.uuid4().hex
    jobs[job_id] = {'status': 'queued', 'result': None, 'error': None}
    def run_job():
        try:
            jobs[job_id]['status'] = 'processing'
            result_name = process_images(source_path, reference_path)
            jobs[job_id]['status'] = 'done'
            jobs[job_id]['result'] = result_name
        except Exception as e:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['error'] = str(e)
    threading.Thread(target=run_job, daemon=True).start()
    return render_template('processing.html', job_id=job_id)

@app.route('/load_model')
def load_model_route():
    """Endpoint to trigger model download/loading before upload."""
    if pipe is None:
        load_model()
        if pipe is None:
            return "模型加载失败，请查看服务器日志。"
        else:
            return "模型已加载完成，您可以上传照片。"
    else:
        return "模型已准备好，无需重新加载。"

@app.route('/status/<job_id>')
def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return {'status': 'unknown'}
    return job

@app.route('/compose')
def compose():
    return render_template('compose.html')

@app.route('/analyze_melody', methods=['POST'])
def analyze_melody():
    # Receive audio file
    if 'audio' not in request.files:
        return {'error': 'No audio provided'}
    audio_file = request.files['audio']
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f'melody_{uuid.uuid4().hex}.wav')
    audio_file.save(audio_path)

    try:
        # Load audio and extract pitch
        y, sr = librosa.load(audio_path, sr=22050)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

        # Extract strongest pitch per frame
        notes = []
        for i in range(pitches.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch = pitches[index, i]
            if pitch > 0:
                note = librosa.hz_to_note(pitch, octave=True)
                notes.append(note)
            else:
                notes.append('rest')

        # Simplify to quarter notes for display
        notes = notes[::20][:16]  # take 16 notes for demo
        return {'notes': [{'pitch': n} for n in notes]}
    except Exception as e:
        return {'error': str(e)}


def process_images(source_path, reference_path):
    # Load model if not loaded
    load_model()
    
    if pipe is None:
        # Fallback to simple blend if model failed
        print("Using simple image blend as fallback")
        source = cv2.imread(source_path)
        reference = cv2.imread(reference_path)
        target_size = (512, 512)
        source = cv2.resize(source, target_size)
        reference = cv2.resize(reference, target_size)
        blended = cv2.addWeighted(source, 0.7, reference, 0.3, 0)
        result_path = os.path.join(app.config['RESULT_FOLDER'], 'result.jpg')
        cv2.imwrite(result_path, blended)
        return 'result.jpg'
    
    # AI-powered hairstyle transfer using ControlNet
    
    # Load images
    source_image = Image.open(source_path).convert("RGB").resize((512, 512))
    reference_image = Image.open(reference_path).convert("RGB").resize((512, 512))
    
    # Create control image from source (canny edges)
    source_np = np.array(source_image)
    source_gray = cv2.cvtColor(source_np, cv2.COLOR_RGB2GRAY)
    source_canny = cv2.Canny(source_gray, 100, 200)
    control_image = Image.fromarray(source_canny).resize((512, 512))
    
    # Prompt for hairstyle change (simplified)
    prompt = "A person with a new modern hairstyle, photorealistic"
    negative_prompt = "blurry, low quality, distorted face"
    
    # Generate new image
    generator = torch.Generator(device="cpu").manual_seed(42)
    output = pipe(
        prompt,
        image=control_image,
        num_inference_steps=20,
        generator=generator,
        negative_prompt=negative_prompt
    )
    
    # Save result
    result_image = output.images[0]
    result_path = os.path.join(app.config['RESULT_FOLDER'], 'result.jpg')
    result_image.save(result_path)
    
    return 'result.jpg'

@app.route('/results/<filename>')
def results(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)