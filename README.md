# Ayushman Mishra
# RA2211027010129
# AE-1

# Real-Time QoS Management for Game Streaming Using Sentiment Analysis and DNN

## Overview

As game streaming platforms like Twitch and YouTube Gaming grow, delivering consistent Quality of Service (QoS) becomes harder due to:

- Varying network conditions  
- Changing stream resolutions and frame rates  
- Fluctuating user engagement  

Traditional QoS methods fail to adapt in real time to user satisfaction signals.

## Objective

To develop a data-driven framework that uses:

- BERT-based sentiment analysis on Twitch reviews  
- Synthetic QoS metrics (bitrate, latency, etc.)  
- Deep Neural Networks (DNNs) for satisfaction classification  

in order to optimize real-time QoS from user-centric feedback.

## Architecture

User Reviews (Twitch) -> BERT Sentiment Analysis -> Add Simulated QoS Metrics -> Noisy Satisfaction Labels -> Balanced Dataset -> DNN Model -> Real-Time Satisfaction Prediction

## Dataset

### 1. Twitch Reviews Dataset
- Around 70,000 app reviews  
- Columns: content, score, thumbsUpCount, etc.  
- Used for user sentiment and feedback

### 2. Simulated QoS Metrics
- Bitrate (2000–8000 kbps)  
- Latency (20–120 ms)  
- Frame rate (30/60 fps)  
- Resolution (480p–1440p)

These mimic the real-world streaming environment.

## Methodology

### 1. Sentiment Analysis
Used `nlptown/bert-base-multilingual-uncased-sentiment` to rate reviews from 1 to 5 stars.

### 2. Noisy Satisfaction Labels
Satisfaction is not directly inferred from sentiment, but with added Gaussian noise:

satisfaction = (sentiment_score + noise >= 4)

This simulates human subjectivity in satisfaction.

### 3. QoS-Driven DNN Model
Trained a 3-layer neural network to classify user satisfaction using:

- sentiment_score  
- bitrate, latency, resolution, frame_rate

Used StandardScaler, train_test_split, and binary_crossentropy loss.

## Results and Visualizations

### Accuracy
- Final DNN model: around 78–85% accuracy
- Generalizes well without overfitting

### Graphs
- Sentiment Score Distribution  
- QoS vs Sentiment (Latency/Bandwidth trends)  
- Correlation Heatmap (insights on what impacts satisfaction)

## Future Improvements

- Incorporate real QoS telemetry from streaming services  
- Use live sentiment from chat/comments  
- Expand with Temporal Neural Networks (RNNs) for session dynamics  
- Build a dashboard for real-time QoS tuning

## Run Instructions

1. Open notebook in Google Colab  
2. Upload `twitch_reviews.csv` from Kaggle  
3. Install dependencies (`transformers`, `tensorflow`)  
4. Run all cells — view plots and evaluation  
5. Modify synthetic QoS ranges to simulate different streaming environments

## Credits

- BERT sentiment model by NLPTown  
- Twitch review data by Ashish Kumar  
- Concept inspired by real-time adaptive streaming and QoE research