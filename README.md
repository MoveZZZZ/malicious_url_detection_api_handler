# 🌐 Malicious URL Detection – API Handler

This project is the **API layer** for the [Malicious URL Detection](https://github.com/MoveZZZZ/malicious_url_detection) and [Malicious URL Detection Plugin](https://github.com/MoveZZZZ/malicious_url_detection_plugin) system.  
It provides a **REST API** interface for serving trained ML/DL models and classifying URLs as **malicious** or **benign**.  

---

## ✨ Features
- 🔗 **Integration with ML Models**  
  - Loads pre-trained models from the detection project  
  - Supports multiple architectures:  
    - Random Forest (RFC)  
    - XGBoost (XGB)  
    - LightGBM (LGBM)  
    - TabNet  
    - DNN (3/5 layers)  
    - Graph Neural Network (GNN)  
    - Autoencoder Classifier (AE)  
    - RBFN + RFC  
    - BERT (URL-based classification)  
    - Stacking Ensemble (meta-model)  

- 🌍 **REST API Endpoints**  
  - `/predict` → classify a single URL  
  - `/batch_predict` → classify multiple URLs  
  - `/health` → check API status  

- ⚡ **Fast Inference**  
  - Uses optimized ML/DL model loading  
  - JSON-based request/response  

- 📝 **Logging**  
  - Logs predictions and errors for monitoring  

---

## 📂 Project Structure
```
malicious_url_detection_api_handler-master/
│── app.py                 # Main API entry point
│── models/                # Saved ML/DL models
│── utils/                 # Helper functions
│── requirements.txt       # Dependencies
│── Dockerfile             # Containerization
│── README.md              # Documentation
```

---

## ⚙️ Requirements
- Python **3.8+**  
- Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Run the API
### 1. Local Run
```bash
python app.py
```

### 2. Docker
```bash
docker build -t malicious-url-api .
docker run -p 8000:8000 malicious-url-api
```

---

## 📡 Example Requests
### Predict single URL
```bash
curl -X POST http://localhost:8000/predict      -H "Content-Type: application/json"      -d '{"url": "http://example-login-secure.com"}'
```

**Response:**
```json
{
  "url": "http://example-login-secure.com",
  "prediction": "phishing",
  "model": "XGBoost",
  "confidence": 0.94
}
```

### Predict batch of URLs
```bash
curl -X POST http://localhost:8000/batch_predict      -H "Content-Type: application/json"      -d '{"urls": ["http://safe.com", "http://paypal-login.xyz"]}'
```

---

## 🛠️ Development
- Language: **Python**  
- Framework: **Flask / FastAPI** (depending on implementation)  
- Models: **RFC, XGB, LGBM, TabNet, DNN, GNN, AE, RBFN+RFC, BERT, Stacking**  

---
