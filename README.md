# Video-Understanding-Toolkit

An implementation of a toolkit for various video processing applications, including action recognition, question answering (VQA), and summarization. This project is inspired by the applied science work on video understanding for AI assistants at Amazon Alexa.

---

## 1. Project Overview

Modern AI assistants are increasingly expected to understand visual data from cameras and video feeds. This project aims to build a suite of tools for analyzing and interpreting video content. The toolkit is broken down into three core applications mentioned in the resume: recognizing actions within a video, answering questions about a video's content, and creating a concise summary of a longer video.

---

## 2. Core Objectives

-   To build a robust pipeline for ingesting and preprocessing video data for deep learning models.
-   To implement a **Video Action Recognition** model to classify the primary activity in a video clip.
-   To create a **Video Question Answering (VQA)** system that can answer natural language questions about a video.
-   To develop a **Video Summarization** model to extract the most important moments from a video.

---

## 3. Methodology

#### Phase 1: Data and Core Preprocessing

1.  **Dataset**: We'll use a standard video dataset that can support multiple tasks. The **MSR-VTT (Microsoft Research Video to Text)** dataset is an excellent choice as it contains video clips with annotations for both categories (actions) and natural language questions/answers.
2.  **Preprocessing**: We'll create a utility script (`/utils/video_loader.py`) that can:
    -   Read a video file using a library like `OpenCV`.
    -   Sample a fixed number of frames from the video clip.
    -   Apply standard image transformations (resizing, normalization) to each frame.
    -   Stack the frames into a tensor, which will be the input to our models.

#### Module 2: Video Action Recognition

1.  **Goal**: Given a short video clip, classify the main action (e.g., "playing guitar," "cooking," "dancing").
2.  **Model**: We'll implement a modern video classification architecture. A great approach is to use a pre-trained **Vision Transformer (ViT)** and adapt it for video.
    -   We'll process each of the sampled frames with the ViT to get frame-level embeddings.
    -   These frame embeddings will be passed to a **GRU or LSTM** layer to capture the temporal sequence.
    -   A final `Linear` layer will classify the output of the GRU into one of the action categories.

#### Module 3: Video Question Answering (VQA)

1.  **Goal**: Given a video and a question in plain English (e.g., "What color is the shirt?"), the model must provide the correct answer.
2.  **Model**: This is a multi-modal task requiring a fusion of video and text information.
    -   **Video Encoder**: We'll use our trained action recognition model (without its final classification layer) to generate a single vector embedding that represents the entire video's content.
    -   **Question Encoder**: We'll use a pre-trained **BERT** model to convert the text question into a vector embedding.
    -   **Fusion and Answering**: The video and question vectors will be concatenated and fed into a simple multi-layer perceptron (MLP). This MLP will act as a classifier, predicting the answer from a predefined set of the most common answers in the dataset.

#### Module 4: Video Summarization

1.  **Goal**: To automatically select the most important frames from a longer video to create a "storyboard" summary. This is known as **extractive summarization**.
2.  **Model**: We'll use an unsupervised approach.
    -   First, we'll divide the input video into short, non-overlapping segments (e.g., 2 seconds each).
    -   We'll use our trained video encoder from the action recognition module to get a feature vector for each segment.
    -   We'll then use a clustering algorithm, like **K-Means**, on these feature vectors. The number of clusters, `k`, will be the number of keyframes we want in our summary.
    -   Finally, we will select the frame closest to the centroid of each cluster as our keyframe. The collection of these `k` keyframes forms the visual summary of the video.

---

## 4. Project Structure
/video-understanding-toolkit
|
|-- /data/
|   |-- get_msr_vtt.sh            # Script to download and prepare the dataset
|
|-- /action_recognition/
|   |-- train_action_classifier.py
|   |-- action_model.py
|
|-- /video_qa/
|   |-- train_vqa_model.py
|   |-- vqa_model.py
|
|-- /video_summarization/
|   |-- generate_summary.py
|
|-- /utils/
|   |-- video_loader.py           # Core video preprocessing and Dataset class
|
|-- app.py                        # A Streamlit app to demo the toolkit
|-- requirements.txt
|-- README.md
