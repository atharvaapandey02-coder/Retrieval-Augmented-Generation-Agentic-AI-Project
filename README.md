<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=220&section=header&text=RAG-Based%20Face%20Recognition%20System&fontSize=36&fontColor=ffffff&animation=fadeIn&fontAlignY=40&desc=Computer%20Vision%20%E2%9C%A6%20RAG%20%E2%9C%A6%20AI%20Reports&descAlignY=65&descSize=18" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-009688?style=for-the-badge)]
[![CLIP](https://img.shields.io/badge/OpenAI-CLIP-412991?style=for-the-badge)]
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?style=for-the-badge&logo=flask&logoColor=white)]
[![Gemini](https://img.shields.io/badge/Google-Gemini%20AI-4285F4?style=for-the-badge&logo=google&logoColor=white)]

<br/>

> **Retrieval-Augmented Generation (RAG) powered Face Recognition with AI-generated Reports**

<br/>

---

</div>

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [📌 Problem Statement](#-problem-statement)
- [🧠 System Architecture](#-system-architecture)
- [⚙️ Methodology](#%EF%B8%8F-methodology)
- [📊 Dataset / Knowledge Base](#-dataset--knowledge-base)
- [📈 Evaluation Metrics](#-evaluation-metrics)
- [🛠️ Tech Stack](#%EF%B8%8F-tech-stack)
- [🚀 Running the Project](#-running-the-project)
- [📂 Project Workflow](#-project-workflow)
- [🔭 Future Improvements](#-future-improvements)

---

## 🎯 Overview

This project implements a **RAG-based Face Recognition System** that identifies celebrities from images and generates **natural language reports**.

It combines:

- **CLIP embeddings** for image understanding  
- **FAISS** for fast similarity search  
- **Google Gemini** for intelligent report generation  

The system not only recognizes faces but also explains results in a human-readable format.

---

## 📌 Problem Statement

Traditional face recognition systems:

- Only return identity labels
- Lack contextual understanding
- Provide no explainability

This project solves these problems using:

- Retrieval-based similarity matching
- Structured knowledge augmentation
- LLM-powered explanation generation

---

## 🧠 System Architecture

```
Input Image
    ↓
CLIP Encoder
    ↓
Image Embedding
    ↓
FAISS Vector Search
    ↓
Top Match Retrieval
    ↓
Knowledge Base Augmentation
    ↓
Google Gemini
    ↓
Natural Language Report
```

### Core Components

- CLIP Image Encoder  
- FAISS Vector Index  
- Structured Knowledge Base  
- Gemini LLM for report generation  
- Flask Web Interface  

---

## ⚙️ Methodology

### 1️⃣ Retrieval

- Convert image → CLIP embedding  
- Perform similarity search using FAISS  
- Retrieve closest match based on threshold  

### 2️⃣ Augmentation

- Fetch structured profile of matched celebrity  
- Add contextual metadata (profession, achievements, etc.)  

### 3️⃣ Generation

- Use Gemini AI to generate:
  - Identity confirmation  
  - Profile summary  
  - Confidence-based explanation  

---

## 📊 Dataset / Knowledge Base

### Context

The system uses a **custom knowledge base of celebrities**.

### Included Personalities

- Actors: Akshay Kumar, Alia Bhatt, Deepika Padukone  
- Cricketers: Virat Kohli, Rohit Sharma  
- Others: Amitabh Bachchan, Kartik Aaryan, etc.  

### Data Components

- Images → stored in `static/enroll/`  
- Metadata → stored in `metadata.json`  
- Vector embeddings → stored in `faiss.index`  

---

## 📈 Evaluation Metrics

- Similarity Score (cosine similarity)
- Match / No-Match decision
- Threshold-based classification
- Qualitative AI-generated report accuracy

---

## 🛠️ Tech Stack

```python
# Deep Learning
PyTorch
OpenCLIP

# Vector Search
FAISS

# Backend
Flask

# AI Generation
Google Gemini API

# Image Processing
Pillow
NumPy
```

---

## 🚀 Running the Project

### Install Dependencies

```
pip install -r requirements.txt
```

### Set API Key

```
export GEMINI_API_KEY="your_api_key_here"
```

Windows:

```
set GEMINI_API_KEY=your_api_key_here
```

### Run Application

```
python app.py
```

Open:

```
http://localhost:5000
```

---

## 📂 Project Workflow

```
Upload Image
        ↓
Generate Embedding (CLIP)
        ↓
Search Similar Faces (FAISS)
        ↓
Retrieve Metadata
        ↓
Generate AI Report (Gemini)
        ↓
Display Results
```

---

## 🔭 Future Improvements

```
📈 Improve accuracy with better embeddings
🧠 Add multi-face detection
🌍 Expand knowledge base
🚀 Deploy using Docker / Cloud
📊 Add real-time webcam recognition
```

---

<div align="center">

**AI-Powered Face Recognition with Explainability using RAG**

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

</div>
