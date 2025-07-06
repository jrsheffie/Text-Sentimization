# Classifying Social Media Content Using Sentiment Analysis  
### IE7500, Summer 2025 - Group 5

## About the Project
Natural Language Processing (NLP) is a branch of artificial intelligence that enables machines to interpret, classify, and generate human language. One key application of NLP is sentiment analysis, which involves breaking down free-form text into tokens for easier manual analysis and categorization. This process allows for valuable insights to be drawn based on the sentiment expressed in the text.  

Today, sentiment analysis is applied across many real-world domains, including but not limited to; gathering product feedback to support user-centered design, measuring audience reactions, capturing live feedback, and conducting competitor analysis. Companies often collect this data from social media comments and incorporate quantitative variables such as the number of likes, shares, and other related metrics. These factors contribute to analyses that support data-driven decision-making.  

This project aims to apply sentiment analysis to data collected from four widely used social media platforms in order to identify which platform, based on this small representative sample, has the highest proportion of positive sentiment. Gaining insight into the overall sentiment of each platform can help inform users who are considering joining a social media community by highlighting the general attitudes of its current users. Also, due to the scalable nature of NLP models, this approach could be retrained on a more tailored dataset such as one specific to a particular brand to analyze and better understand customer sentiment toward that brand.  

We expect to identify clear distinctions in sentiment across major social media platforms Facebook, Instagram, Twitter, and TikTok by analyzing user-generated content using Natural Language Processing techniques. Our primary goal is to classify captions, and posts as either positive or negative based on sentiment analysis and then calculate the ratio of positive to negative posts for each platform. This will allow us to rank the platforms from most to least positive, as well as from most to least negative. To evaluate the accuracy of our model, we will apply standard validation metrics. Additionally, we will manually review a sample of posts to ensure that sentiment classifications are accurate and not based on faulty assumptions. Ultimately, we aim to provide a data-driven comparison of sentiment trends across platforms and draw insights into which platforms tend to foster more positive or negative online discourse.  

The insights generated from this work are not only valuable for understanding current sentiment dynamics but also serve as a foundation for future research and decision-making. For businesses, this data-driven comparison may inform where to focus community engagement or advertising efforts based on the tone of discourse. For platform designers and policymakers, the results could highlight areas for improvement in fostering healthier, more positive online environments.  

In conclusion, this project contributes to a deeper understanding of how sentiment varies across major social media platforms, providing a meaningful snapshot of digital community sentiment and a foundation for ongoing exploration into the emotional climate of social media.  

## Getting Started
tk



## Data 
This dataset is a one-week snapshot (December 1–7, 2024) of global online discourse, compiled and released by [Exorde Labs](https://exorde.network). It contains approximately **65.5 million posts** from public sources including social media platforms, blogs, and news websites.

### 🔍 Key Features

- **Scale & Scope:** ~65.5 million entries from 6,000+ public sources (e.g., X, Reddit, Bluesky, YouTube, Mastodon).
- **Multilingual Coverage:** Content spans **122 languages**.
- **Rich Metadata:**
  - Sentiment score (ranging from –1 to +1)
  - Main emotion label
  - Primary theme (e.g., Politics, Business, Entertainment)
  - Extracted English keywords via KeyBERT
- **Timestamps:** All posts include UTC timestamps.
- **Privacy-Conscious:** Author identities are anonymized using SHA-1 hashing.
[Exorde Social Media December 2024 - Week 1 Dataset](https://huggingface.co/datasets/Exorde/exorde-social-media-december-2024-week1)

We also used the below code to clean out our data out of HTTP Urls, @ mentions using regex, and emoji symbols using regex patterns.
Also made sure to extract letters and trim any spaces in our data.

```
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|@\w+|#\w+|[^\x00-\x7F]+", " ", text)  # Removing url using regex on HTTP, @ regex mentions, and emoji code regex patterns
    text = re.sub(r"[^a-z\s]", "", text)                          # Letter Only Regex
    text = re.sub(r"\s+", " ", text).strip()                     # Space Trim Logic
    return text
```

The slice of data we're using can be found [here](https://github.com/jrsheffie/Text-Sentimization/tree/main/Dataset) in our github repo

## Contributors
- Marta Herzog
- Josiah Sheffie
- Tamara Bonheur
- Shyam Patel



