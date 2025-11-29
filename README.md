# 📰 FlipItNews - NLP Multi-Class Text Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.51.0-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**An advanced NLP system for automated news article classification using Word Embeddings and Machine Learning**

[Live Demo](https://flipitnews-nlp-classifier-md5fomvan7qnwpq7keylem.streamlit.app/) • [Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Documentation](#-documentation)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Models & Performance](#-models--performance)
- [Screenshots](#-screenshots)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)
- [Acknowledgments](#-acknowledgments)

---
🚀 Live Demo

**Streamlit Profile**: View on Streamlit Cloud(https://flipitnews-nlp-classifier-md5fomvan7qnwpq7keylem.streamlit.app/)

---

## 🎯 Overview

**FlipItNews** is a comprehensive NLP project that automatically classifies news articles into 5 categories: **Technology**, **Business**, **Sports**, **Entertainment**, and **Politics**. The system leverages advanced text preprocessing, multiple machine learning algorithms, and Word2Vec embeddings to achieve **91.24% accuracy**.

This project includes:
- ✅ Complete data preprocessing pipeline
- ✅ 7 different ML models (Naive Bayes, SGD, Logistic Regression, etc.)
- ✅ Word2Vec embeddings with FastText
- ✅ Interactive Streamlit dashboard
- ✅ Real-time activity logging
- ✅ Comprehensive visualizations

---

## 🎓 Problem Statement

FlipItNews aims to revolutionize financial literacy by providing personalized, categorized news content. The challenge is to:

1. **Automatically classify** news articles into relevant categories
2. **Achieve high accuracy** (>90%) for reliable content delivery
3. **Process text efficiently** using NLP techniques
4. **Provide insights** through interactive visualizations
5. **Enable real-time predictions** for new articles

---

## ✨ Features

### 🔍 Data Processing
- **Stopwords Removal** - Eliminates common words
- **Punctuation Cleaning** - Removes special characters
- **Stemming & Lemmatization** - Reduces words to root forms
- **Text Normalization** - Lowercasing and standardization

### 🤖 Machine Learning Models
- **Naive Bayes** - Probabilistic classifier
- **SGD Classifier** - Linear SVM with decision scores
- **Logistic Regression** - Best performer (91.24% accuracy)
- **Decision Tree** - Tree-based classification
- **Random Forest** - Ensemble method
- **K-Nearest Neighbors** - Instance-based learning
- **Word2Vec + Logistic Regression** - Deep learning embeddings

### 📊 Interactive Dashboard
- **6 Interactive Tabs**:
  1. 📊 Dataset Overview - Statistics and distributions
  2. 🔄 Data Processing - Text transformation pipeline
  3. 🤖 Model Results - Performance comparison
  4. 📈 Visualizations - Charts and graphs
  5. 🔮 Predictions - Real-time classification
  6. 📝 Activity Log - User interaction tracking

### 🎨 UI Features
- Beautiful gradient design (purple/blue theme)
- Real-time model training with progress bars
- Confusion matrices and classification reports
- Confidence scores and decision functions
- Sample articles for testing
- Downloadable logs

---

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.11** - Programming language
- **Jupyter Notebook** - Development environment
- **Streamlit** - Web application framework

### Machine Learning & NLP
- **Scikit-learn** - ML algorithms and pipelines
- **NLTK** - Natural language processing
- **Spacy** - Advanced NLP (en_core_web_sm)
- **Gensim** - Word2Vec embeddings (fasttext-wiki-news-subwords-300)

### Data & Visualization
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib** - Plotting library
- **Seaborn** - Statistical visualizations

---

## 📁 Project Structure

```
NLP_Word_Embedding_Word2Vec/
│
├── 📊 Data
│   └── flipitnews-data.csv                          # Dataset (2,225 articles)
│
├── 📓 Notebooks
│   ├── FLIPLTNews_Word_Embedding_Word2Vec_2.ipynb  # Main analysis notebook
│   └── FlipItNews_Word_Embedding_Word2Vec_1.ipynb  # Initial exploration
│
├── 🐍 Python Scripts
│   ├── app.py                                       # Streamlit dashboard
│   └── FLIPLTNews_Word_Embedding_Word2Vec_2.py     # Converted notebook script
│
├── 📄 Documentation
│   ├── README.md                                    # This file
│   ├── LICENSE                                      # MIT License
│   ├── CONTRIBUTING.md                              # Contribution guidelines
│   ├── execution_log.txt                            # Execution history
│   └── NLP FlipIt News.txt                          # Project notes
│
├── 📑 Case Studies & Reports
│   ├── Business Case _ NLP FlipItNews Approach.pdf # Business case
│   ├── FlipItNews_Case_Study_1.pdf                 # Detailed case study
│   └── flipitnews-word-embedding-word2vec_2.pdf    # Technical report
│
├── 🔧 Configuration
│   ├── requirements.txt                             # Python dependencies
│   └── .gitignore                                   # Git ignore rules
│
└── 📸 Assets
    └── screenshots/                                 # UI screenshots
```

---

## 🚀 Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/Ratnesh-181998/NLP-Word-Embedding-FlipItNews.git
cd NLP-Word-Embedding-FlipItNews
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Spacy Model
```bash
python -m spacy download en_core_web_sm
```

### Step 5: Download NLTK Data
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

---

## 💻 Usage

### Running the Streamlit Dashboard

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Running the Jupyter Notebook

```bash
jupyter notebook FLIPLTNews_Word_Embedding_Word2Vec_2.ipynb
```

### Running the Python Script

```bash
python FLIPLTNews_Word_Embedding_Word2Vec_2.py
```

---

## 📈 Models & Performance

### Model Comparison

| Model | Accuracy | Best Category | F1-Score |
|-------|----------|---------------|----------|
| **Logistic Regression + Word2Vec** | **91.24%** | Sports | 0.97 |
| Logistic Regression | 90.56% | Sports | 0.96 |
| Naive Bayes | 88.31% | Technology | 0.92 |
| SGD Classifier | 87.64% | Sports | 0.94 |
| Random Forest | 85.39% | Business | 0.89 |
| Decision Tree | 82.47% | Entertainment | 0.86 |
| K-Nearest Neighbors | 79.33% | Politics | 0.82 |

### Best Model Performance (Logistic Regression + Word2Vec)

```
               precision    recall  f1-score   support

     Business       0.85      0.93      0.89       102
Entertainment       0.96      0.88      0.92        77
     Politics       0.92      0.82      0.87        84
       Sports       0.95      0.98      0.97       102
   Technology       0.90      0.93      0.91        80

     accuracy                           0.91       445
    macro avg       0.92      0.91      0.91       445
 weighted avg       0.91      0.91      0.91       445
```

### Dataset Statistics

- **Total Articles**: 2,225
- **Categories**: 5 (Technology, Business, Sports, Entertainment, Politics)
- **Train/Test Split**: 80/20
- **Vocabulary Size**: ~15,000 unique words
- **Average Article Length**: 450 characters

**Category Distribution**:
- Sports: 511 articles (23%)
- Business: 510 articles (23%)
- Politics: 417 articles (19%)
- Technology: 401 articles (18%)
- Entertainment: 386 articles (17%)

---

## 📸 Screenshots

### Dashboard Overview
![Dashboard](screenshots/dashboard_overview.png)
<img width="2836" height="1472" alt="image" src="https://github.com/user-attachments/assets/f3fd122e-edfa-4587-a024-37e712ab2944" />
<img width="2828" height="1452" alt="image" src="https://github.com/user-attachments/assets/d610a2cc-22a2-4a1d-8df1-9256d324d6e8" />
<img width="2844" height="1440" alt="image" src="https://github.com/user-attachments/assets/0dbd7b38-c197-49f6-9478-8e03cd0351d9" />
<img width="2869" height="1414" alt="image" src="https://github.com/user-attachments/assets/2670eef3-c694-4acd-888c-3e1d9078e608" />
<img width="2831" height="1425" alt="image" src="https://github.com/user-attachments/assets/ab724d0c-f618-4e0d-b8ae-af4d2633223e" />
<img width="2822" height="1407" alt="image" src="https://github.com/user-attachments/assets/fd4d18d1-76d9-41e6-8f2c-42e5dac00417" />
<img width="2816" height="1398" alt="image" src="https://github.com/user-attachments/assets/c047fb75-94c3-4a52-b26d-095002369398" />

### Model Results
![Model Results](screenshots/model_results.png)
<img width="2869" height="1408" alt="image" src="https://github.com/user-attachments/assets/30aa9013-a8ea-47cd-8621-a314fed0028f" />
<img width="2841" height="1472" alt="image" src="https://github.com/user-attachments/assets/7419b10a-d2da-4787-9d6a-5d31006deaa2" />

### Predictions
![Predictions](screenshots/predictions.png)
<img width="2863" height="1469" alt="image" src="https://github.com/user-attachments/assets/795a7956-31fd-474f-9214-95d8700f4725" />
<img width="2748" height="1484" alt="image" src="https://github.com/user-attachments/assets/832f8f50-be47-4974-b178-35ed3d1574f6" />
<img width="2856" height="1418" alt="image" src="https://github.com/user-attachments/assets/bd7ffbcf-dc2c-4c3d-90b4-a26af954f163" />
<img width="2774" height="1447" alt="image" src="https://github.com/user-attachments/assets/0ea04ee6-5f6b-46c2-983b-1d70b4711365" />
<img width="2794" height="1455" alt="image" src="https://github.com/user-attachments/assets/e57b0db4-7119-4210-87a2-37887f55408a" />
<img width="2819" height="1462" alt="image" src="https://github.com/user-attachments/assets/10be74a5-6efa-4893-9f00-2f32fef69a7c" />
<img width="2808" height="1472" alt="image" src="https://github.com/user-attachments/assets/b81d94a1-a8e8-432a-9c80-f8ef7dfc5636" />
<img width="2856" height="1435" alt="image" src="https://github.com/user-attachments/assets/13456454-1ba6-41d2-8d0c-0606c3d56aca" />

### Activity Log
![Activity Log](screenshots/activity_log.png)
<img width="2848" height="1447" alt="image" src="https://github.com/user-attachments/assets/94fdacbe-1971-4592-b636-65c81a0c39ee" />
<img width="2866" height="1398" alt="image" src="https://github.com/user-attachments/assets/76e43435-44e5-42cf-8af7-f8575e81cde0" />

---

## 📚 Documentation

### Key Files

1. **`app.py`** - Streamlit dashboard with 6 interactive tabs
2. **`FLIPLTNews_Word_Embedding_Word2Vec_2.ipynb`** - Complete analysis notebook
3. **`flipitnews-data.csv`** - News articles dataset
4. **`execution_log.txt`** - Detailed execution history with timestamps

### PDF Documentation

- **Business Case**: Comprehensive business analysis and approach
- **Case Study**: Detailed technical implementation and results
- **Technical Report**: In-depth methodology and findings

### Code Structure

#### Main Components

**Data Preprocessing** (`app.py` lines 110-145):
- Stopwords removal
- Punctuation cleaning
- Stemming and lemmatization
- Text normalization

**Model Training** (`app.py` lines 147-217):
- Pipeline creation
- TF-IDF vectorization
- Model fitting and evaluation
- Caching for performance

**Prediction System** (`app.py` lines 435-543):
- Real-time classification
- Confidence scores
- Decision function visualization
- Sample article testing

**Activity Logging** (`app.py` lines 30-42, 520-605):
- User interaction tracking
- Timestamp recording
- Log visualization
- Export functionality

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Ratnesh Kumar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 📧 Contact

**Ratnesh Kumar**

- 📧 Email: [rattudacsit2021gate@gmail.com](mailto:rattudacsit2021gate@gmail.com)
- 💼 LinkedIn: [https://www.linkedin.com/in/ratneshkumar1998/](https://www.linkedin.com/in/ratnesh-kumar-181998)
- 🐙 GitHub: [@Ratnesh-181998](https://github.com/Ratnesh-181998)
- 📱 Portfolio: [ratnesh-kumar.dev](https://github.com/Ratnesh-181998)

---

## 🙏 Acknowledgments

- **FlipItNews** for the business case and dataset
- **Scikit-learn** team for excellent ML libraries
- **Streamlit** for the amazing web framework
- **Gensim** for Word2Vec embeddings
- **NLTK & Spacy** for NLP tools
- Open source community for continuous support

---

## 🎓 Learning Outcomes

This project demonstrates:

✅ **NLP Fundamentals** - Text preprocessing, tokenization, stemming, lemmatization  
✅ **Machine Learning** - Multiple algorithms, hyperparameter tuning, model evaluation  
✅ **Deep Learning** - Word embeddings, Word2Vec, FastText  
✅ **Web Development** - Streamlit dashboard, interactive UI  
✅ **Data Visualization** - Matplotlib, Seaborn, confusion matrices  
✅ **Software Engineering** - Code structure, documentation, version control  
✅ **Performance Optimization** - Caching, parallel processing, efficient algorithms  

---

## 📊 Project Metrics

- **Lines of Code**: ~1,500
- **Development Time**: 50 hours
- **Models Trained**: 7
- **Accuracy Achieved**: 91.24%
- **Dataset Size**: 2,225 articles
- **Features Engineered**: 5,000 (TF-IDF)

---

## 🔮 Future Enhancements

- [ ] Deploy to cloud (Heroku/AWS/GCP)
- [ ] Add more news categories
- [ ] Implement BERT/Transformer models
- [ ] Create REST API
- [ ] Add multi-language support
- [ ] Integrate with news APIs
- [ ] Build mobile app
- [ ] Add user authentication
- [ ] Implement A/B testing
- [ ] Create recommendation system

---

<div align="center">

**⭐ If you found this project helpful, please give it a star!**

Made with ❤️ by [Ratnesh Kumar](https://github.com/Ratnesh-181998)

</div>
