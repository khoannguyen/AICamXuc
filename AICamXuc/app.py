from flask import Flask, render_template, request, jsonify
import os
import time
from werkzeug.utils import secure_filename
from predict import predict_emotion

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'data/saveaudio'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'wav'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_data' not in request.files:
        return jsonify({'error': 'Không có file gửi lên'}), 400
    f = request.files['audio_data']
    if not (f and allowed_file(f.filename)):
        return jsonify({'error': 'Chỉ chấp nhận file .wav'}), 400

    ts = time.strftime("%Y%m%d-%H%M%S")
    base = secure_filename(f.filename.rsplit('.',1)[0])
    fn = f"{base}_{ts}.wav"
    path = os.path.join(app.config['UPLOAD_FOLDER'], fn)
    f.save(path)

    try:
        buckets = predict_emotion(path)
    except Exception as e:
        return jsonify({'error': f'Xử lý âm thanh thất bại: {e}'}), 500

    return jsonify(buckets)

if __name__ == '__main__':
    app.run(debug=True)
