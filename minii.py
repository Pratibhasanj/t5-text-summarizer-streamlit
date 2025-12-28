import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer # type: ignore

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_name = "t5-small"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Summarization function
def summarize_text(text, max_length=150, min_length=50):
    input_text = "summarize: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    summary_ids = model.generate(
        input_ids, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# App title and description
st.title("📝 Text Summarization with T5 Transformer")
st.markdown("Welcome! This app summarizes text using a T5 Transformer model. Enter text or upload a file, then generate your summary. 🤖")

# Theme toggle
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #f9f9f9;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
    }
    .download-button > button {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True
)

# Initialize session state for text input, summary, and history
if "user_text" not in st.session_state:
    st.session_state["user_text"] = ""
if "summary" not in st.session_state:
    st.session_state["summary"] = ""
if "summary_history" not in st.session_state:
    st.session_state["summary_history"] = []

# Text input box
st.session_state["user_text"] = st.text_area("📋 Enter text to summarize:", value=st.session_state["user_text"], height=200)

# File uploader
uploaded_file = st.file_uploader("📂 Or upload a text file:", type="txt")
if uploaded_file is not None:
    st.session_state["user_text"] = uploaded_file.read().decode("utf-8")

# Character count display
st.write(f"Character count: {len(st.session_state['user_text'])}")

# Generate summary button
if st.button("✨ Generate Summary"):
    if st.session_state["user_text"].strip():
        with st.spinner("Summarizing... Please wait."):
            summary = summarize_text(st.session_state["user_text"])
            st.session_state["summary"] = summary
            st.session_state["summary_history"].append(summary)
    else:
        st.warning("⚠️ Please enter some text or upload a file to summarize.")

# Display summary if available
if st.session_state["summary"]:
    st.subheader("📝 Summary:")
    st.write(st.session_state["summary"])

    # Download summary button
    st.download_button(
        label="📥 Download Summary",
        data=st.session_state["summary"],
        file_name="summary.txt",
        mime="text/plain",
        key="download-button",
    )

# History of previous summaries
if st.session_state["summary_history"]:
    st.sidebar.subheader("🔍 Summary History")
    for idx, prev_summary in enumerate(st.session_state["summary_history"], 1):
        with st.sidebar.expander(f"Summary {idx}"):
            st.write(prev_summary)

# Clear button
if st.button("❌ Clear"):
    st.session_state["user_text"] = ""
    st.session_state["summary"] = ""
    st.session_state["summary_history"] = []