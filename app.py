import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import os

# Download NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

# Logging function
def log_interaction(action, details=""):
    """Log user interactions to execution_log.txt"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"\n[{timestamp}] UI Interaction: {action}"
    if details:
        log_entry += f" - {details}"
    
    try:
        with open("execution_log.txt", "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        pass  # Silently fail if logging doesn't work

# Page configuration
st.set_page_config(
    page_title="FlipItNews NLP Analysis",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #667eea;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 2rem;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📰 FlipItNews NLP Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### *By Ratnesh Kumar - Word Embedding & Classification*")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/news.png", width=100)
    st.markdown("## 🎯 Project Overview")
    st.info("""
    **FlipItNews** aims to revolutionize financial literacy through AI/ML.
    
    This project categorizes news articles into:
    - 📱 Technology
    - 💼 Business
    - ⚽ Sports
    - 🎬 Entertainment
    - 🏛️ Politics
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    
# Initialize session state for app start logging
if 'app_started' not in st.session_state:
    st.session_state.app_started = True
    log_interaction("App Started", "User opened the Streamlit dashboard")
    
# Load and process data
@st.cache_data
def load_data():
    df = pd.read_csv('flipitnews-data.csv')
    return df

@st.cache_data
def preprocess_data(df):
    # Remove stopwords
    def remove_stopwords(text):
        clean_text = ' '.join([i for i in text.split() if i not in stopwords.words('english')])
        return clean_text
    
    # Remove punctuation
    def remove_punctuation(text):
        cleantext = ''.join([i for i in text if i not in string.punctuation])
        return cleantext
    
    # Stemming
    ps = PorterStemmer()
    def stemming(text):
        clean_text = ' '.join([ps.stem(i) for i in text.split()])
        return clean_text
    
    # Lemmatization
    wl = WordNetLemmatizer()
    def lemmatize(text):
        clean_text = ' '.join([wl.lemmatize(i) for i in text.split()])
        return clean_text
    
    df_processed = df.copy()
    df_processed['Article'] = df_processed['Article'].apply(remove_stopwords)
    df_processed['Article'] = df_processed['Article'].apply(remove_punctuation)
    df_processed['Article'] = df_processed['Article'].str.lower()
    df_processed['Article'] = df_processed['Article'].apply(stemming)
    df_processed['Article'] = df_processed['Article'].apply(lemmatize)
    
    # Encode categories
    le = LabelEncoder()
    df_processed['Category_id'] = le.fit_transform(df_processed['Category'])
    
    return df_processed, le

@st.cache_data(show_spinner=False)
def train_models(df_processed):
    X = df_processed['Article']
    y = df_processed['Category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    
    # Create a progress bar
    progress_text = "Training models... Please wait."
    progress_bar = st.progress(0, text=progress_text)
    
    # Naive Bayes
    progress_bar.progress(10, text="Training Naive Bayes...")
    nb = Pipeline([
        ('vect', CountVectorizer(max_features=5000)),  # Limit features for speed
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    results['Naive Bayes'] = {
        'model': nb,
        'accuracy': accuracy_score(y_test, y_pred_nb),
        'report': classification_report(y_test, y_pred_nb, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred_nb),
        'y_test': y_test,
        'y_pred': y_pred_nb
    }
    
    # SGD Classifier
    progress_bar.progress(40, text="Training SGD Classifier...")
    sgd = Pipeline([
        ('vect', CountVectorizer(max_features=5000)),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=100, tol=1e-3)),
    ])
    sgd.fit(X_train, y_train)
    y_pred_sgd = sgd.predict(X_test)
    results['SGD Classifier'] = {
        'model': sgd,
        'accuracy': accuracy_score(y_test, y_pred_sgd),
        'report': classification_report(y_test, y_pred_sgd, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred_sgd),
        'y_test': y_test,
        'y_pred': y_pred_sgd
    }
    
    # Logistic Regression
    progress_bar.progress(70, text="Training Logistic Regression...")
    logreg = Pipeline([
        ('vect', CountVectorizer(max_features=5000)),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression(n_jobs=-1, C=1e5, max_iter=500)),  # Use all cores
    ])
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)
    results['Logistic Regression'] = {
        'model': logreg,
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'report': classification_report(y_test, y_pred_lr, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred_lr),
        'y_test': y_test,
        'y_pred': y_pred_lr
    }
    
    progress_bar.progress(100, text="Training complete!")
    progress_bar.empty()
    
    return results, X_test, y_test

# Load data
try:
    df = load_data()
    df_processed, le = preprocess_data(df)
    
    # Sidebar stats
    with st.sidebar:
        st.metric("Total Articles", len(df))
        st.metric("Categories", df['Category'].nunique())
        st.metric("Vocabulary Size", len(set(' '.join(df_processed['Article']).split())))
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dataset Overview", 
        "🔄 Data Processing", 
        "🤖 Model Results",
        "📈 Visualizations",
        "🔮 Predictions",
        "📝 Activity Log"
    ])
    
    with tab1:
        log_interaction("Viewed Dataset Overview tab")
        st.markdown('<p class="sub-header">Dataset Overview</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card"><h3>Total Articles</h3><h1>{}</h1></div>'.format(len(df)), unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><h3>Categories</h3><h1>{}</h1></div>'.format(df['Category'].nunique()), unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><h3>Features</h3><h1>{}</h1></div>'.format(len(df.columns)), unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Sample Data")
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Category Distribution")
            category_counts = df['Category'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b']
            category_counts.plot(kind='bar', ax=ax, color=colors)
            ax.set_xlabel('Category', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Articles per Category', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()
            
            # Show counts
            st.dataframe(category_counts.to_frame('Count'), use_container_width=True)
    
    with tab2:
        log_interaction("Viewed Data Processing tab")
        st.markdown('<p class="sub-header">Data Processing Pipeline</p>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🔧 Text Processing Steps:
        1. **Stopwords Removal** - Remove common words (the, is, at, etc.)
        2. **Punctuation Removal** - Clean special characters
        3. **Lowercasing** - Convert all text to lowercase
        4. **Stemming** - Reduce words to root form
        5. **Lemmatization** - Convert words to dictionary form
        """)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📝 Original Text")
            sample_idx = st.selectbox("Select article index:", range(min(10, len(df))))
            log_interaction("Selected article for comparison", f"Index: {sample_idx}")
            st.text_area("Original", df.iloc[sample_idx]['Article'], height=200)
        
        with col2:
            st.markdown("#### ✨ Processed Text")
            st.text_area("Processed", df_processed.iloc[sample_idx]['Article'], height=200)
        
        st.markdown("---")
        st.markdown("#### 📊 Processed Dataset")
        st.dataframe(df_processed.head(10), use_container_width=True)
    
    with tab3:
        log_interaction("Viewed Model Results tab")
        st.markdown('<p class="sub-header">Model Performance Comparison</p>', unsafe_allow_html=True)
        
        # Train models (will use cache after first run)
        results, X_test, y_test = train_models(df_processed)
        
        st.info("✅ Models trained and cached! Results will load instantly on subsequent visits.")
        
        # Model comparison
        st.markdown("### 🏆 Model Accuracy Comparison")
        
        accuracies = {name: res['accuracy'] for name, res in results.items()}
        
        col1, col2, col3 = st.columns(3)
        for idx, (name, acc) in enumerate(accuracies.items()):
            with [col1, col2, col3][idx]:
                st.markdown(f'<div class="metric-card"><h4>{name}</h4><h2>{acc:.2%}</h2></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed results for each model
        for model_name, model_results in results.items():
            with st.expander(f"📊 {model_name} - Detailed Results", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Classification Report")
                    report_df = pd.DataFrame(model_results['report']).transpose()
                    st.dataframe(report_df.style.background_gradient(cmap='RdYlGn', subset=['f1-score']), use_container_width=True)
                
                with col2:
                    st.markdown("#### Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(model_results['confusion_matrix'], 
                               annot=True, 
                               fmt='d', 
                               cmap='Blues',
                               xticklabels=df['Category'].unique(),
                               yticklabels=df['Category'].unique(),
                               ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title(f'{model_name} - Confusion Matrix')
                    st.pyplot(fig)
                    plt.close()
    
    with tab4:
        log_interaction("Viewed Visualizations tab")
        st.markdown('<p class="sub-header">Data Visualizations</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Model Accuracy Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            models = list(accuracies.keys())
            accs = list(accuracies.values())
            colors = ['#667eea', '#764ba2', '#f093fb']
            bars = ax.barh(models, accs, color=colors)
            ax.set_xlabel('Accuracy', fontsize=12)
            ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
            ax.set_xlim([0, 1])
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{width:.2%}', 
                       ha='left', va='center', fontsize=10, fontweight='bold')
            
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("#### 🎯 Category-wise Performance")
            # Get best model
            best_model_name = max(accuracies, key=accuracies.get)
            best_model_report = results[best_model_name]['report']
            
            categories = [cat for cat in best_model_report.keys() if cat not in ['accuracy', 'macro avg', 'weighted avg']]
            f1_scores = [best_model_report[cat]['f1-score'] for cat in categories]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(categories, f1_scores, color=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b'])
            ax.set_ylabel('F1-Score', fontsize=12)
            ax.set_title(f'{best_model_name} - Category Performance', fontsize=14, fontweight='bold')
            ax.set_ylim([0, 1])
            plt.xticks(rotation=45)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2%}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # Article length distribution
        st.markdown("#### 📏 Article Length Distribution")
        df['article_length'] = df['Article'].str.len()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        for category in df['Category'].unique():
            category_data = df[df['Category'] == category]['article_length']
            ax.hist(category_data, alpha=0.5, label=category, bins=30)
        
        ax.set_xlabel('Article Length (characters)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Article Length Distribution by Category', fontsize=14, fontweight='bold')
        ax.legend()
        st.pyplot(fig)
        plt.close()
    
    with tab5:
        log_interaction("Viewed Predictions tab")
        st.markdown('<p class="sub-header">Make Predictions</p>', unsafe_allow_html=True)
        
        st.markdown("### 🔮 Try the News Classifier")
        
        # Select model
        model_choice = st.selectbox("Select Model:", list(results.keys()))
        log_interaction("Selected model for prediction", f"Model: {model_choice}")
        selected_model = results[model_choice]['model']
        
        # Input text
        user_input = st.text_area(
            "Enter a news article:",
            height=200,
            placeholder="Paste or type a news article here..."
        )
        
        if st.button("🚀 Classify Article", type="primary"):
            if user_input:
                log_interaction("Classified article", f"Model: {model_choice}, Text length: {len(user_input)} chars")
                with st.spinner("Analyzing..."):
                    prediction = selected_model.predict([user_input])[0]
                    log_interaction("Prediction result", f"Category: {prediction}")
                    
                    st.success(f"### Predicted Category: **{prediction}**")
                    
                    # Check if model supports predict_proba
                    if hasattr(selected_model, 'predict_proba'):
                        try:
                            prediction_proba = selected_model.predict_proba([user_input])[0]
                            
                            st.markdown("#### Confidence Scores:")
                            
                            # Create confidence chart
                            categories = selected_model.classes_
                            fig, ax = plt.subplots(figsize=(10, 6))
                            colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b']
                            bars = ax.barh(categories, prediction_proba, color=colors)
                            ax.set_xlabel('Confidence', fontsize=12)
                            ax.set_title('Prediction Confidence by Category', fontsize=14, fontweight='bold')
                            ax.set_xlim([0, 1])
                            
                            # Add value labels
                            for bar, prob in zip(bars, prediction_proba):
                                width = bar.get_width()
                                ax.text(width, bar.get_y() + bar.get_height()/2, 
                                       f'{prob:.2%}', 
                                       ha='left', va='center', fontsize=10, fontweight='bold')
                            
                            st.pyplot(fig)
                            plt.close()
                        except:
                            st.info("ℹ️ This model doesn't provide confidence scores. Try Naive Bayes or Logistic Regression for probability estimates.")
                    elif hasattr(selected_model, 'decision_function'):
                        # For SGD Classifier, show decision function scores
                        try:
                            decision_scores = selected_model.decision_function([user_input])[0]
                            categories = selected_model.classes_
                            
                            st.markdown("#### Decision Scores:")
                            st.info("💡 SGD Classifier uses decision scores instead of probabilities. Higher scores indicate stronger confidence.")
                            
                            # Normalize scores to 0-1 range for visualization
                            import numpy as np
                            scores_normalized = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min() + 1e-10)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b']
                            bars = ax.barh(categories, scores_normalized, color=colors)
                            ax.set_xlabel('Normalized Decision Score', fontsize=12)
                            ax.set_title('Decision Scores by Category', fontsize=14, fontweight='bold')
                            ax.set_xlim([0, 1])
                            
                            # Add value labels with original scores
                            for bar, score, norm_score in zip(bars, decision_scores, scores_normalized):
                                width = bar.get_width()
                                ax.text(width, bar.get_y() + bar.get_height()/2, 
                                       f'{score:.2f}', 
                                       ha='left', va='center', fontsize=10, fontweight='bold')
                            
                            st.pyplot(fig)
                            plt.close()
                            
                            # Show raw scores in a table
                            scores_df = pd.DataFrame({
                                'Category': categories,
                                'Decision Score': decision_scores,
                                'Normalized': scores_normalized
                            }).sort_values('Decision Score', ascending=False)
                            
                            st.dataframe(scores_df.style.background_gradient(cmap='RdYlGn', subset=['Decision Score']), use_container_width=True)
                        except Exception as e:
                            st.info("ℹ️ This model doesn't provide confidence scores. Try Naive Bayes or Logistic Regression for probability estimates.")
                    else:
                        st.info("ℹ️ This model doesn't provide confidence scores. Try Naive Bayes or Logistic Regression for probability estimates.")
            else:
                st.warning("Please enter some text to classify.")
        
        st.markdown("---")
        st.markdown("### 📝 Sample Articles to Try:")
        
        sample_articles = {
            "Technology": "Apple unveils new iPhone with advanced AI features and improved camera system. The latest smartphone includes breakthrough technology for enhanced user experience.",
            "Business": "Stock markets rally as major companies report strong quarterly earnings. Investors show confidence in economic recovery amid positive financial indicators.",
            "Sports": "Manchester United defeats Liverpool in thrilling Premier League match. The game ended 3-2 with a last-minute goal securing the victory.",
            "Entertainment": "New blockbuster movie breaks box office records on opening weekend. The film starring top Hollywood actors receives critical acclaim.",
            "Politics": "Parliament debates new legislation on climate change policy. Government officials discuss measures to reduce carbon emissions and promote renewable energy."
        }
        
        for category, article in sample_articles.items():
            if st.button(f"Try {category} Sample"):
                log_interaction("Loaded sample article", f"Category: {category}")
                st.text_area("Sample Article:", article, height=100, key=f"sample_{category}")
    
    with tab6:
        log_interaction("Viewed Activity Log tab")
        st.markdown('<p class="sub-header">Activity Log</p>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 📝 Real-time Interaction Tracking
        This tab shows all your interactions with the dashboard. Logs are automatically saved to `execution_log.txt`.
        """)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("#### Recent Activity")
        
        with col2:
            if st.button("🔄 Refresh Logs"):
                st.rerun()
        
        # Read and display logs
        try:
            if os.path.exists("execution_log.txt"):
                with open("execution_log.txt", "r", encoding="utf-8") as f:
                    log_content = f.read()
                
                # Extract UI interaction logs
                ui_logs = []
                for line in log_content.split('\n'):
                    if 'UI Interaction:' in line:
                        ui_logs.append(line)
                
                if ui_logs:
                    st.markdown("---")
                    
                    # Display in reverse order (most recent first)
                    recent_logs = ui_logs[-50:][::-1]  # Last 50 logs, reversed
                    
                    for log in recent_logs:
                        # Parse log entry
                        if '[' in log and ']' in log:
                            timestamp = log[log.find('[')+1:log.find(']')]
                            message = log[log.find(']')+1:].strip()
                            
                            # Color code based on action
                            if 'Viewed' in message:
                                icon = "👁️"
                                color = "#667eea"
                            elif 'Selected' in message:
                                icon = "🎯"
                                color = "#764ba2"
                            elif 'Classified' in message or 'Prediction' in message:
                                icon = "🚀"
                                color = "#43e97b"
                            elif 'Loaded' in message:
                                icon = "📄"
                                color = "#f093fb"
                            elif 'Started' in message:
                                icon = "🎬"
                                color = "#4facfe"
                            else:
                                icon = "📌"
                                color = "#999"
                            
                            st.markdown(f"""
                            <div style='padding: 0.5rem; margin: 0.3rem 0; border-left: 3px solid {color}; background: rgba(102, 126, 234, 0.05); border-radius: 5px;'>
                                <span style='color: {color}; font-size: 1.2rem;'>{icon}</span>
                                <strong style='color: #666; font-size: 0.85rem;'>{timestamp}</strong><br/>
                                <span style='margin-left: 1.5rem;'>{message}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.info(f"📊 Showing {len(recent_logs)} most recent interactions")
                else:
                    st.info("No UI interactions logged yet. Start exploring the dashboard!")
                
                # Download button for full log
                st.markdown("#### 💾 Download Full Log")
                st.download_button(
                    label="Download execution_log.txt",
                    data=log_content,
                    file_name=f"flipitnews_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            else:
                st.warning("Log file not found. Interactions will be logged as you use the app.")
        except Exception as e:
            st.error(f"Error reading log file: {str(e)}")

except FileNotFoundError:
    st.error("❌ Error: 'flipitnews-data.csv' file not found. Please ensure the file is in the same directory as this app.")
except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>📰 <strong>FlipItNews NLP Analysis Dashboard</strong></p>
    <p>Developed by Ratnesh Kumar | Word Embedding & Multi-Class Classification</p>
    <p>Using: Scikit-learn, NLTK, Streamlit, Pandas, Matplotlib, Seaborn</p>
</div>
""", unsafe_allow_html=True)
