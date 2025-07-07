# Classifying Social Media Content Using Sentiment Analysis  
### IE7500, Summer 2025 - Group 5

## About the Project

This project uses Natural Language Processing (NLP) to perform sentiment analysis on user-generated content from four major social media platforms — Facebook, Instagram, TikTok, and X (formerly Twitter).
The goal is to classify posts from each platform as **positive** or **negative**, then compare sentiment ratios to determine which platform exhibits the most positive overall tone. Special attention is given to X (Twitter) to analyze its sentiment trends and how they compare to other platforms.
By tokenizing and classifying text data, validating results with standard metrics and manual review, we aim to provide a data-driven snapshot of sentiment across platforms. This work offers insights into user attitudes and lays the groundwork for future brand- or topic-specific sentiment studies.

## Model Development

Among several models explored, we used Logistic Regression to analyze text sentiment as an initial approach due to its simplicity, interpretability, and effectiveness in binary classification—making it well-suited for identifying positive,neutral, and negative sentiment in refrence to online discourse across topics.



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



