# 📐 Code Structure Documentation

This document provides a detailed overview of the codebase structure, architecture, and key components.

---

## 📁 Directory Structure

```
NLP_Word_Embedding_Word2Vec/
│
├── 📊 Data Layer
│   └── flipitnews-data.csv                          # 2,225 news articles
│       ├── Columns: Article, Category
│       ├── Size: ~5 MB
│       └── Categories: Technology, Business, Sports, Entertainment, Politics
│
├── 📓 Analysis & Development
│   ├── FLIPLTNews_Word_Embedding_Word2Vec_2.ipynb  # Main notebook (781 KB)
│   │   ├── Data exploration
│   │   ├── Preprocessing pipeline
│   │   ├── Model training (7 models)
│   │   ├── Word2Vec implementation
│   │   └── Evaluation & visualization
│   │
│   └── FlipItNews_Word_Embedding_Word2Vec_1.ipynb  # Initial exploration (778 KB)
│       └── Preliminary analysis
│
├── 🐍 Application Layer
│   ├── app.py                                       # Streamlit dashboard (29 KB)
│   │   ├── UI Components (6 tabs)
│   │   ├── Model training pipeline
│   │   ├── Prediction system
│   │   ├── Visualization engine
│   │   └── Activity logging
│   │
│   └── FLIPLTNews_Word_Embedding_Word2Vec_2.py     # Converted script (23 KB)
│       └── Batch processing version
│
├── 📄 Documentation Layer
│   ├── README.md                                    # Project overview
│   ├── LICENSE                                      # MIT License
│   ├── CONTRIBUTING.md                              # Contribution guidelines
│   ├── CODE_STRUCTURE.md                            # This file
│   ├── execution_log.txt                            # Execution history (13 KB)
│   └── NLP FlipIt News.txt                          # Project notes (3.7 KB)
│
├── 📑 Reports & Case Studies
│   ├── Business Case _ NLP FlipItNews Approach.pdf # Business analysis (111 KB)
│   ├── FlipItNews_Case_Study_1.pdf                 # Technical case study (924 KB)
│   └── flipitnews-word-embedding-word2vec_2.pdf    # Research report (597 KB)
│
└── 🔧 Configuration
    ├── requirements.txt                             # Python dependencies
    ├── .gitignore                                   # Git ignore rules
    └── .venv/                                       # Virtual environment
```

---

## 🏗️ Application Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                       │
│  ┌──────┬──────┬──────┬──────┬──────┬──────┐               │
│  │ Data │ Proc │ Model│ Viz  │ Pred │ Log  │               │
│  └──────┴──────┴──────┴──────┴──────┴──────┘               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Business Logic Layer                      │
│  ┌──────────────┬──────────────┬──────────────┐            │
│  │ Preprocessing│ Model Training│ Prediction   │            │
│  └──────────────┴──────────────┴──────────────┘            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                       Data Layer                             │
│  ┌──────────────┬──────────────┬──────────────┐            │
│  │ CSV Data     │ Cached Models│ Logs         │            │
│  └──────────────┴──────────────┴──────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📝 File-by-File Breakdown

### 1. `app.py` - Streamlit Dashboard (Main Application)

**Purpose**: Interactive web application for news classification

**Structure**:
```python
# Lines 1-42: Imports & Configuration
├── Standard libraries (pandas, numpy, matplotlib)
├── ML libraries (sklearn, nltk, spacy, gensim)
├── Streamlit configuration
└── Logging function

# Lines 43-103: UI Configuration
├── Page config (title, icon, layout)
├── Custom CSS (gradients, colors, styling)
├── Header and sidebar
└── Project overview

# Lines 104-145: Data Processing Functions
├── load_data() - Load CSV with caching
└── preprocess_data() - Text preprocessing pipeline
    ├── remove_stopwords()
    ├── remove_punctuation()
    ├── stemming()
    └── lemmatize()

# Lines 147-217: Model Training
└── train_models() - Train 3 ML models
    ├── Naive Bayes pipeline
    ├── SGD Classifier pipeline
    ├── Logistic Regression pipeline
    └── Progress bar tracking

# Lines 219-250: Main Application Logic
├── Load and process data
├── Display sidebar statistics
└── Create 6 tabs

# Lines 252-290: Tab 1 - Dataset Overview
├── Metrics cards (articles, categories, features)
├── Sample data table
├── Category distribution chart
└── Category counts

# Lines 292-318: Tab 2 - Data Processing
├── Processing steps description
├── Original vs processed text comparison
├── Article selector
└── Processed dataset preview

# Lines 320-363: Tab 3 - Model Results
├── Model training (cached)
├── Accuracy comparison cards
├── Detailed results per model
├── Classification reports
└── Confusion matrices

# Lines 365-433: Tab 4 - Visualizations
├── Model accuracy bar chart
├── Category-wise F1-scores
└── Article length distribution

# Lines 435-519: Tab 5 - Predictions
├── Model selector
├── Text input area
├── Classify button
├── Prediction display
├── Confidence scores (Naive Bayes, Logistic Regression)
├── Decision scores (SGD Classifier)
└── Sample articles

# Lines 521-607: Tab 6 - Activity Log
├── Log viewer
├── Color-coded entries
├── Refresh button
├── Download button
└── Real-time updates

# Lines 609-620: Error Handling & Footer
├── Exception handling
└── Footer with credits
```

**Key Features**:
- **Caching**: `@st.cache_data` for performance
- **Progress Bars**: Visual feedback during training
- **Logging**: Real-time interaction tracking
- **Responsive Design**: Adaptive layout

---

### 2. `FLIPLTNews_Word_Embedding_Word2Vec_2.ipynb` - Analysis Notebook

**Purpose**: Complete NLP analysis and model development

**Structure**:
```
Cell 1-5: Introduction & Setup
├── Problem statement
├── Dataset description
├── Import libraries
└── Load data

Cell 6-15: Exploratory Data Analysis
├── Dataset shape and info
├── Category distribution
├── Missing values check
├── Sample articles
└── Statistical summary

Cell 16-30: Text Preprocessing
├── Stopwords removal
├── Punctuation cleaning
├── Lowercasing
├── Stemming (PorterStemmer)
├── Lemmatization (WordNetLemmatizer)
└── Before/after comparison

Cell 31-45: Feature Engineering
├── Bag of Words (CountVectorizer)
├── TF-IDF (TfidfVectorizer)
├── Train-test split (80/20)
└── Feature matrix creation

Cell 46-70: Model Training & Evaluation
├── Naive Bayes
│   ├── Pipeline creation
│   ├── Training
│   ├── Predictions
│   └── Evaluation (accuracy, classification report)
│
├── SGD Classifier
│   ├── Pipeline with TF-IDF
│   ├── Training
│   └── Evaluation
│
├── Logistic Regression
│   ├── Pipeline creation
│   ├── Hyperparameter tuning
│   └── Evaluation
│
├── Decision Tree
├── Random Forest
└── K-Nearest Neighbors

Cell 71-90: Word2Vec Implementation
├── Load FastText embeddings (958 MB)
├── Create sentence vectors
├── PCA visualization (2D)
├── Train Logistic Regression on embeddings
└── Final evaluation (91.24% accuracy)

Cell 91-100: Results & Conclusion
├── Model comparison table
├── Best model selection
├── Confusion matrix visualization
├── Key insights
└── Future improvements
```

---

### 3. `FLIPLTNews_Word_Embedding_Word2Vec_2.py` - Python Script

**Purpose**: Batch processing version of the notebook

**Structure**:
```python
# Lines 1-50: Imports & Setup
# Lines 51-100: Data Loading & EDA
# Lines 101-200: Preprocessing Functions
# Lines 201-400: Model Training (7 models)
# Lines 401-500: Word2Vec Implementation
# Lines 501-667: Evaluation & Results
```

---

## 🔧 Key Components

### 1. Data Preprocessing Pipeline

```python
def preprocess_data(df):
    """
    Complete text preprocessing pipeline
    
    Steps:
    1. Remove stopwords (NLTK)
    2. Remove punctuation
    3. Lowercase conversion
    4. Stemming (PorterStemmer)
    5. Lemmatization (WordNetLemmatizer)
    6. Label encoding
    
    Returns:
        Processed DataFrame with cleaned text
    """
    # Implementation...
```

**Input**: Raw text articles  
**Output**: Cleaned, normalized text  
**Libraries**: NLTK, string, sklearn

---

### 2. Model Training Pipeline

```python
def train_models(df_processed):
    """
    Train multiple ML models with TF-IDF
    
    Models:
    1. Naive Bayes (MultinomialNB)
    2. SGD Classifier (Linear SVM)
    3. Logistic Regression
    
    Returns:
        Dictionary with model results
    """
    # Pipeline: CountVectorizer → TF-IDF → Classifier
```

**Pipeline Structure**:
```
Raw Text → CountVectorizer → TF-IDF → Classifier → Prediction
```

---

### 3. Prediction System

```python
def predict_article(text, model):
    """
    Classify a news article
    
    Args:
        text: Article text
        model: Trained pipeline
        
    Returns:
        - Predicted category
        - Confidence scores (if available)
        - Decision scores (for SGD)
    """
```

---

### 4. Activity Logging System

```python
def log_interaction(action, details=""):
    """
    Log user interactions to file
    
    Format: [YYYY-MM-DD HH:MM:SS] UI Interaction: Action - Details
    
    Tracked Events:
    - App startup
    - Tab navigation
    - Model selection
    - Predictions
    - Sample loading
    """
```

---

## 📊 Data Flow

### Training Flow

```
CSV Data
    ↓
Load & Validate
    ↓
Preprocessing
    ├── Stopwords Removal
    ├── Punctuation Cleaning
    ├── Stemming
    └── Lemmatization
    ↓
Feature Extraction
    ├── CountVectorizer (5000 features)
    └── TF-IDF Transform
    ↓
Train/Test Split (80/20)
    ↓
Model Training
    ├── Naive Bayes
    ├── SGD Classifier
    └── Logistic Regression
    ↓
Evaluation
    ├── Accuracy
    ├── Precision/Recall/F1
    └── Confusion Matrix
    ↓
Cache Models
```

### Prediction Flow

```
User Input
    ↓
Preprocessing (same pipeline)
    ↓
Vectorization (TF-IDF)
    ↓
Model Prediction
    ↓
Post-processing
    ├── Category Label
    ├── Confidence Scores
    └── Decision Scores
    ↓
Display Results
```

---

## 🎨 UI Components

### Tab Structure

```
Dashboard
│
├── Tab 1: Dataset Overview
│   ├── Metrics (cards)
│   ├── Sample data (table)
│   └── Distribution (chart)
│
├── Tab 2: Data Processing
│   ├── Pipeline description
│   ├── Text comparison
│   └── Processed data
│
├── Tab 3: Model Results
│   ├── Accuracy cards
│   ├── Classification reports
│   └── Confusion matrices
│
├── Tab 4: Visualizations
│   ├── Model comparison
│   ├── Category performance
│   └── Article length dist.
│
├── Tab 5: Predictions
│   ├── Model selector
│   ├── Text input
│   ├── Results display
│   └── Sample articles
│
└── Tab 6: Activity Log
    ├── Log viewer
    ├── Refresh button
    └── Download button
```

---

## 🔄 State Management

### Streamlit Session State

```python
# App initialization
if 'app_started' not in st.session_state:
    st.session_state.app_started = True
    log_interaction("App Started")

# Caching
@st.cache_data  # Data caching
@st.cache_data(show_spinner=False)  # Silent caching
```

---

## 📈 Performance Optimizations

1. **Caching**
   - Data loading: `@st.cache_data`
   - Model training: Cached after first run
   - Preprocessing: Cached results

2. **Feature Limitation**
   - CountVectorizer: max_features=5000
   - Reduces vocabulary size
   - 3-5x faster training

3. **Parallel Processing**
   - Logistic Regression: n_jobs=-1
   - Uses all CPU cores

4. **Progress Indicators**
   - Visual feedback during training
   - Better UX

---

## 🧪 Testing Strategy

### Manual Testing
- UI interaction testing
- Model prediction testing
- Edge case testing

### Validation
- Cross-validation (80/20 split)
- Confusion matrix analysis
- Classification reports

---

## 📦 Dependencies

### Core (Required)
- streamlit >= 1.51.0
- pandas >= 2.1.4
- scikit-learn >= 1.4.0
- nltk >= 3.8.1
- spacy >= 3.8.11
- gensim >= 4.4.0

### Visualization
- matplotlib >= 3.8.2
- seaborn >= 0.13.1

### Optional
- jupyter (for notebook development)

---

## 🔐 Security Considerations

- No sensitive data in repository
- Environment variables for API keys (if needed)
- Input validation for user text
- Safe file operations

---

## 🚀 Deployment Considerations

### Local Deployment
```bash
streamlit run app.py
```

### Cloud Deployment (Future)
- Heroku
- AWS EC2
- Google Cloud Run
- Streamlit Cloud

---

## 📝 Code Conventions

- **PEP 8** compliance
- **Type hints** where applicable
- **Docstrings** for all functions
- **Comments** for complex logic
- **Modular design** for reusability

---

## 🔮 Future Enhancements

1. **API Development**
   - REST API with FastAPI
   - Endpoint for predictions

2. **Model Improvements**
   - BERT/Transformer models
   - Ensemble methods
   - Hyperparameter optimization

3. **UI Enhancements**
   - Dark mode toggle
   - Export predictions
   - Batch processing

4. **Monitoring**
   - Performance metrics
   - Error tracking
   - Usage analytics

---

**Last Updated**: 2025-11-28  
**Version**: 1.0.0  
**Maintainer**: Ratnesh Kumar
