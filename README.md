# Classifying Social Media Content Using Sentiment Analysis  
### IE7500, Summer 2025 - Group 5

## 📚 Table of Contents

- [Classifying Social Media Content Using Sentiment Analysis](#classifying-social-media-content-using-sentiment-analysis)
  - [About the Project](#about-the-project)
  - [Setup](#setup)
  - [Model Development](#model-development)
  - [Dataset](#dataset)
  - [Contributors](#contributors)

## About the Project

Natural Language Processing (NLP) is a branch of artificial intelligence that enables machines to interpret, classify, and generate human language. One key application of NLP is sentiment analysis, which involves breaking down free-form text into tokens for easier manual analysis and categorization. This process allows for valuable insights to be drawn based on the sentiment expressed in the text.  

Today, sentiment analysis is applied across many real-world domains, including but not limited to: gathering product feedback to support user-centered design, measuring audience reactions, capturing live feedback and conducting competitor analysis. Companies often collect this data from social media comments and incorporate quantitative variables such as the number of likes, shares, and other related metrics. These factors contribute to analyses that support data-driven decision-making.  

This project aims to apply sentiment analysis to data collected from the social media platform X in order to identify which posts have a positive or negative sentiment. Gaining insight into the sentiment of content published on X can serve as an extrapolation of the overall attitudes which users of the platform have.  


## Setup

All code can be run using the provided jupyter notebook scripts in the Models folder with the related dataset in our dataset folder.
For setting up the package dependencies for your system please refer to [requirements](https://github.com/jrsheffie/Text-Sentimization/blob/main/requirements.txt)

## Model Development

## 🧠 Model Architectures & Training Procedures

This project applies several machine learning and deep learning models to perform sentiment analysis on social media text data. The models range from traditional algorithms to state-of-the-art transformer-based models.

---

### 🔍 Model Choices

#### 1. Logistic Regression (Baseline)
- **Why**: Simple, interpretable, and computationally efficient.
- **Preprocessing**: Text is vectorized using TF-IDF.
- **Strength**: Fast training and a good baseline for comparison.

#### 2. Feedforward Neural Network (FFNN)
- **Why**: A non-linear model that captures more complex patterns than Logistic Regression.
- **Architecture**:
  - Dense layers with ReLU activation
  - Dropout layers for regularization
  - Sigmoid or softmax output layer depending on the task
- **Training**:
  - Loss Function: Binary Cross-Entropy
  - Optimizer: Adam
  - Input: TF-IDF vectors or word embeddings

#### 3. Long Short-Term Memory (LSTM)
- **Why**: Designed to handle sequential data and retain context over time.
- **Architecture**:
  - Embedding Layer (pretrained or trainable)
  - One or more LSTM layers
  - Dropout and Dense layers
- **Training**:
  - Loss: Binary Cross-Entropy
  - Optimizer: Adam
  - Handles padded sequences

#### 4. BERT (Bidirectional Encoder Representations from Transformers)
- **Why**: Pretrained on massive corpora, BERT excels at understanding contextual meaning in text.
- **Setup**:
  - Uses HuggingFace `transformers` library
  - Tokenized using `bert-base-uncased` tokenizer
  - Fine-tuned for classification by adding a dense head on top of the [CLS] token
- **Training**:
  - Loss: CrossEntropyLoss
  - Optimizer: AdamW
  - Learning rate scheduler and warm-up steps applied

---

### ⚙️ Training Procedures

- **Preprocessing**:
  - Lowercasing, punctuation removal, tokenization
  - TF-IDF vectorization for classical models
  - Word embeddings or BERT tokenizer for deep models
  - Padding for LSTM and BERT input sequences

- **Data Split**:
  - 80/20 or 70/30 train-test split
  - Stratified sampling to maintain class balance

- **Metrics Used**:
  - Accuracy
  - F1 Score (macro/weighted)
  - Confusion Matrix for qualitative analysis

- **Callbacks & Techniques**:
  - `EarlyStopping` to avoid overfitting
  - `ReduceLROnPlateau` to adjust learning rate
  - Validation monitoring

- **Environment**:
  - Deep learning models trained using GPU acceleration (Google Colab or CUDA environment)


---




## Dataset

**Multi-Source, Multi-Language Social Media Dataset (1 Week Sample)**  
This dataset offers a rich, high-resolution snapshot of global digital discourse, collected over the period of **December 1 to December 7, 2024**, and curated by [Exorde Labs](https://www.exordelabs.com/). The sample provided here includes **5,000 English-language posts** sourced primarily from X.com (formerly Twitter), showcasing the thematic and emotional range of global online conversations.

This subset is ideal for tasks such as **sentiment analysis**, **emotion detection**, **thematic classification**, and **social media trend analysis**.

📂 Full dataset available: [Hugging Face – Exorde December 2024 Week 1](https://huggingface.co/datasets/Exorde/exorde-social-media-december-2024-week1)

---

### Dataset Highlights

- **Time-Stamped**: Each post is recorded with an exact UTC timestamp at the moment of posting.
- **English-Language Sample**: This subset focuses on posts detected as English (`language="en"`).
- **High-Quality Annotations**:
  - Sentiment score (range: -1 to +1)
  - Emotion label (26 emotion classes)
  - Primary & secondary topic themes (e.g., Technology, Politics)
  - English keyword extraction
- **Privacy-Preserving**: User identifiers are hashed (SHA-1) or omitted when unavailable.
- **Cleaned Content**: Each entry includes raw and cleaned versions of the text.

---

### Dataset Schema

| Column Name       | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `date`            | ISO timestamp of post (UTC)                                                 |
| `original_text`   | Raw text content as posted                                                 |
| `cleaned_text`    | Lowercased, preprocessed version of `original_text`                        |
| `url`             | Original post URL                                                           |
| `author_hash`     | SHA-1 hash of author ID (null for anonymous)                                |
| `language`        | ISO 639-1 language code (always `"en"` in this sample)                      |
| `primary_theme`   | Main classified topic from 15 thematic categories                           |
| `secondary_themes`| List of numeric theme codes for secondary topics                            |
| `english_keywords`| Keywords extracted via KeyBERT and statistical algorithms                   |
| `sentiment`       | Float from -1 (negative) to 1 (positive)                                    |
| `main_emotion`    | Emotion classification (e.g., *neutral*, *realization*, *approval*)         |

---

### Sample Statistics (5000 English posts)

- **Date Range**: 2024-12-01 to 2024-12-07  
- **Top Themes**: Technology, People, Politics, Sports  
- **Most Common Emotions**: *Neutral*, *Realization*, *Approval*  
- **Sentiment Range**: -0.89 to +0.91  
- **Average Sentiment**: ~0.05  

---

### Use Cases

This dataset enables a variety of research and development scenarios, including:

- 📊 **Real-time sentiment and trend tracking**  
- 🧠 **Emotion detection across events**  
- 🔍 **Thematic analysis for news and social media**  
- 🧾 **Text classification and NLP model benchmarking**  
- 🛰 **Global discourse monitoring**

---

### Citation

If using this dataset, please cite as:

**Exorde Labs. (2024). Multi-Source, Multi-Language Social Media Dataset (1 Week Sample) [Data set].**  
[https://www.exordelabs.com](https://www.exordelabs.com)



### Data Cleansing

We used the following below to cleanse our data from Removing URLs, cleaning mentions, and removing emoji codes.
Also isolated it only to the platform X.
```
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|@\w+|#\w+|[^\x00-\x7F]+", " ", text)  # Removing url using regex on HTTP, @ regex mentions, and emoji code regex patterns
    text = re.sub(r"[^a-z\s]", "", text)                          # Letter Only Regex
    text = re.sub(r"\s+", " ", text).strip()                     # Space Trim Logic
    return text

en_ds = ds['train'].filter(lambda example: example['language'] == 'en' and example['url'].startswith('https://x.com/') and clean_text(example.get('original_text', '')) != '')
sample_en_ds = en_ds.shuffle(seed=42).select(range(min(5000, len(en_ds))))
sample_en_ds = sample_en_ds.map(lambda example: {**example, 'cleaned_text': clean_text(example.get('original_text', ''))})
sample_en_df = sample_en_ds.to_pandas()
sample_en_df
```

You can find the jupyter notebook containing our cleaning logic [here](https://github.com/jrsheffie/Text-Sentimization/blob/main/Models/dataset_cleaning.ipynb)
The slice of data we're using can be found [here](https://github.com/jrsheffie/Text-Sentimization/tree/main/Dataset) in our github repo

## Contributors
- Marta Herzog
- Josiah Sheffie
- Tamara Bonheur
- Shyam Patel



