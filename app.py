from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PREDICTION_FOLDER'] = 'static/predictions/'

# Modeli yükle
model = YOLO('best.pt')  # Model dosyan bu dizinde olmalı

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'Dosya bulunamadı.', 400
    file = request.files['file']
    if file.filename == '':
        return 'Dosya seçilmedi.', 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # YOLO tahmin
    results = model.predict(filepath, save=True, project='static', name='predictions', exist_ok=True)

    # Biyofilik skor için sınıf sayımı
    class_labels = model.names  # {0: 'bitki', 1: 'doğal', ...}
    score_mapping = {'bitki': 1, 'doğal': 1, 'hayvan': 1, 'pencere': 1}

    total_score = 0
    element_counts = {}

    # Her sınıfı sadece 1 kez saymak için bir küme oluşturuyoruz
    counted_classes = set()

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            label = class_labels[cls_id]

            if label in score_mapping and label not in counted_classes:
                # Bu sınıfı ilk kez görüyoruz
                element_counts[label] = 1
                total_score += score_mapping[label]
                counted_classes.add(label)

    # Skoru normalize et (örnek: 0-100 arası)
    normalized_score = min(total_score * 10, 100)  # Ağırlıklı skor, 100 ile sınırlı

    # Akıllı öneri sistemi
    suggestions = []
    if 'bitki' not in element_counts:
        suggestions.append("Bitkiler, biyofilik ortamı güçlendirir. Birkaç bitki ekleyebilirsiniz.")
    if 'hayvan' not in element_counts:
        suggestions.append("Bir hayvan sahiplenmek biyofilik ortamı zenginleştirebilir.")
    if 'pencere' not in element_counts:
        suggestions.append("Doğal ışık alan bir pencere açmanız önerilir.")
    if 'doğal' not in element_counts:
        suggestions.append("Doğal malzemeli (ahşap, taş gibi) duvar veya yüzeyler inşa edebilirsiniz.")

    # Son mesaj: Biyofilik mi değil mi?
    biyofilik_mesaj = "Bu oda biyofilik olarak değerlendirildi! 🌿" if normalized_score >= 30 else "Bu oda biyofilik değildir. 🛑"

    return render_template('result.html',
                           filename=filename,
                           score=round(normalized_score, 2),
                           elements=element_counts,
                           suggestions=suggestions,
                           biyofilik_mesaj=biyofilik_mesaj)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PREDICTION_FOLDER'], exist_ok=True)
    app.run(debug=True)
