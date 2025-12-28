# T5 Text Summarizer (Streamlit App)

This project is an AI-based text summarization application built using the **T5 Transformer model** and **Streamlit**. It allows users to input text or upload a text file and generates a concise summary using a pretrained NLP model.

## Features
- Text summarization using `t5-small`
- Input via text box or file upload
- Downloadable summary output
- Summary history tracking
- Simple and user-friendly Streamlit interface

## Tech Stack
- Python
- Hugging Face Transformers
- Streamlit
- PyTorch

## How It Works
The application prepends the input text with a summarization prompt and passes it to a pretrained T5 model. The generated output is decoded into a readable summary and displayed to the user.

## Running the Project
Due to local system memory limitations, this project is intended to be run on cloud-based environments such as:
- Google Colab
- Streamlit Cloud
- Systems with sufficient RAM for transformer models

Basic steps:
```bash
pip install streamlit transformers torch
streamlit run app.py
