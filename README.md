[Eren KIRMIZI Bitirme Raporu.pdf](https://github.com/user-attachments/files/23140918/Eren.KIRMIZI.Bitirme.Raporu.pdf)


# Automated Biophilic Element Detection

## 1. Project Description
Bu proje, görsellerde **biophilic tasarım öğelerini** tespit eder.  
YOLO modeli ile nesne algılama yapılır ve Flask tabanlı backend üzerinden web arayüzüne sunulur.

## 2. System Architecture
**Model Pipeline:**  
Dataset → Training → Model Export (.pt) → Inference  

**Web Pipeline:**  
Frontend (HTML/JS) ↔ Flask Backend ↔ YOLO Model  

**Veri akışı:**  
- Kullanıcı görsel yükler  
- Flask backend, YOLO modelini çağırır  
- Algılanan öğeler görsel üzerine işlenir ve UI’ya geri döner

## 3. Technologies Used
- Python  
- YOLO  
- Flask  
- HTML / CSS / JS (Frontend)

## 4. Installation & Setup
```bash
git clone https://github.com/Eren-KIRMIZI/Automated-Recognition-and-Analysis-of-Biophilic-Design-Elements-Using-AI.git
cd project-folder

# Virtual environment
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# YOLO model dosyasını weights/ içine koyun
