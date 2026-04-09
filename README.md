# SeizureAura: AI-Based Seizure Risk Prediction System

An end-to-end AI system for detecting aura stages and predicting seizure risk using NLP and EEG-based analysis.

---

## 🚀 Overview

SeizureAura is designed to assist in early detection of seizure risk by combining symptom-based analysis with predictive modeling.

The system integrates:
- AI chatbot for symptom intake  
- NLP pipeline for aura-stage classification  
- EEG-based seizure risk prediction  

Designed as a unified pipeline for transforming patient inputs into actionable risk insights.

---

## 🎯 Problem

Epilepsy patients often experience early warning signs (auras), but:
- Symptoms are subjective and difficult to track  
- Early prediction is inconsistent  
- No unified system combines symptom analysis and EEG signals  

---

## 💡 Solution

SeizureAura provides an integrated system that:
- Collects symptoms via chatbot  
- Classifies aura stages using NLP  
- Predicts seizure risk using EEG-based models  

---

## 🧠 Key Features

- AI chatbot for symptom intake  
- NLP-based aura stage detection  
- EEG-based seizure risk prediction  
- Flask REST APIs for model inference  
- End-to-end integrated pipeline  

---

## ⚙️ System Workflow

1. User inputs symptoms through chatbot  
2. NLP pipeline processes and classifies aura stage  
3. EEG-based model evaluates seizure risk  
4. Backend generates prediction output  
5. Results are returned to the user  

---

## 📁 Project Structure

```bash
seizureaura-app/
├── backend/
│   ├── app.py
│   ├── knowledge_base/
│   ├── models/              # (Not included due to size constraints)
│   ├── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
│
└── README.md
```
🧩 Tech Stack
Python
Flask
NLP (Text Processing)
Machine Learning
React / Frontend UI

## ⚠️ Note on Model Files
Due to GitHub size limitations, trained model files are not included in this repository.

The system is designed to load models externally or during runtime, and the implementation for model integration is fully provided.

⚡ API Overview
POST /predict → Predict seizure risk
POST /analyze → Detect aura stage from symptoms

📸 Demo
Homepage
<img width="1906" height="948" alt="01_Seizure_Aura_Homepage" src="https://github.com/user-attachments/assets/107103ea-3670-4c7e-a7d6-0950094ccbb3" />

Chatbot Interaction
<img width="858" height="794" alt="03_AI_Chatbot_Knowledge_Response_UI" src="https://github.com/user-attachments/assets/0d3a235e-7b1e-4ee9-ae42-357f50eba24b" />

Aura Stage Detection
<img width="716" height="576" alt="06_Aura_Stage_Analysis_Output2" src="https://github.com/user-attachments/assets/9c911b57-07ea-4a9f-bebc-b2a5affa0995" />

Seizure Risk Prediction
<img width="701" height="534" alt="08_EEG_Seizure_Risk_Output" src="https://github.com/user-attachments/assets/95be64b2-5048-4efe-adcc-fb72910b8647" />

📈 Highlights
Combines symptom analysis and EEG-based prediction
Demonstrates real-world healthcare AI use case
End-to-end system from input to prediction

🚧 Limitations
Requires structured symptom input
No real-time EEG integration
