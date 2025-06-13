const fileInput   = document.getElementById('fileInput');
const fileBtn     = document.getElementById('fileInputBtn');
const recordBtn   = document.getElementById('recordBtn');
const predictBtn  = document.getElementById('predictBtn');
const fileNameSpn = document.getElementById('fileName');
const audioPlayer = document.getElementById('audioPlayer');
const resultsDiv  = document.getElementById('results');
const resultLabel = document.getElementById('resultLabel');
const resultProb  = document.getElementById('resultProb');
const resultTime  = document.getElementById('resultTime');
const historyBody = document.getElementById('historyBody');
const timerSpan   = document.getElementById('recordTimer');

let recorder, stream, currentBlob, historyCount = 0;
let autoStopTimer, timerInterval, seconds = 0, isRecording = false;

// Helpers
function resetResult() {
  resultsDiv.style.display = 'block';
  resultsDiv.textContent = 'Vui lòng cung cấp dữ liệu âm thanh.';
  resultLabel.style.display = resultProb.style.display = resultTime.style.display = 'none';
}

// File select
fileBtn.onclick = () => fileInput.click();
fileInput.onchange = () => {
  if (!fileInput.files.length) return;
  currentBlob = fileInput.files[0];
  fileNameSpn.textContent = currentBlob.name;
  audioPlayer.src = URL.createObjectURL(currentBlob);
  resetResult();
};

// Recording toggle
recordBtn.onclick = async () => {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
};

function startRecording() {
  navigator.mediaDevices.getUserMedia({ audio: true })
    .then(s => {
      stream = s;
      recorder = RecordRTC(stream, {
        type: 'audio',
        mimeType: 'audio/wav',
        recorderType: StereoAudioRecorder,
        desiredSampRate: 16000,
      });
      recorder.startRecording();
      isRecording = true;
      recordBtn.textContent = 'Dừng ghi âm';
      recordBtn.classList.add('recording');
      startTimer();
      autoStopTimer = setTimeout(stopRecording, 30000);
      resetResult();
    })
    .catch(err => alert('Không thể truy cập micro: ' + err));
}

function stopRecording() {
  recorder.stopRecording(() => {
    currentBlob = recorder.getBlob();
    const name = 'record_' + new Date().toLocaleTimeString() + '.wav';
    fileNameSpn.textContent = name;
    audioPlayer.src = URL.createObjectURL(currentBlob);
    recorder = null;
  });
  stream.getTracks().forEach(t => t.stop());
  isRecording = false;
  recordBtn.textContent = 'Ghi âm trực tiếp';
  recordBtn.classList.remove('recording');
  clearTimeout(autoStopTimer);
  stopTimer();
}

// Timer functions
function startTimer() {
  seconds = 0;
  timerSpan.textContent = `0s`;
  timerInterval = setInterval(() => {
    seconds++;
    timerSpan.textContent = `${seconds}s`;
  }, 1000);
}
function stopTimer() {
  clearInterval(timerInterval);
  timerSpan.textContent = '';
}

// Predict
predictBtn.onclick = async () => {
  if (!currentBlob) {
    alert('Vui lòng ghi âm hoặc chọn file .wav trước khi nhận diện.');
    return;
  }
  predictBtn.disabled = true;
  predictBtn.innerHTML = '<span class="spinner"></span>Đang xử lý';

  const form = new FormData();
  form.append('audio_data', currentBlob, 'audio.wav');

  try {
    const res = await fetch('/predict', { method: 'POST', body: form });
    const data = await res.json();

    // Flatten & pick top
    let flat = [];
    Object.values(data).forEach(arr => arr.forEach(([l,p]) => flat.push({label:l, prob:p})));
    const top = flat.reduce((a,b)=> b.prob>a.prob?b:a, flat[0]);

    // Show result
    resultsDiv.style.display = 'none';
    resultLabel.style.display = resultProb.style.display = resultTime.style.display = 'block';
    resultLabel.textContent = top.label;
    resultProb.textContent  = 'Tỉ lệ: ' + (top.prob*100).toFixed(2) + '%';
    const now = new Date().toLocaleTimeString();
    resultTime.textContent  = 'Cập nhật lúc ' + now;

    // Add to history
    historyCount++;
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${now}</td><td>${top.label}</td><td>${(top.prob*100).toFixed(2)}%</td>`;
    historyBody.prepend(tr);

  } catch (err) {
    alert('Lỗi khi nhận diện: ' + err.message);
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = 'Nhận diện';
  }
};

// Init
resetResult();
