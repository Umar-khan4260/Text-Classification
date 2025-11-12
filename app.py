import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Page config
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="wide"
)

# Cache the model loading
@st.cache_resource
def load_model():
    try:
        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", 
            num_labels=3,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, "./model")
        tokenizer = AutoTokenizer.from_pretrained("./model")
        
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
            
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Label mapping
id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}

def predict_sentiment(text, model, tokenizer):
    if not text.strip():
        return None, None
    
    try:
        # Tokenize input
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        pred_id = probs.argmax()
        return id2label[pred_id], probs
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# Main app
def main():
    st.title("🎭 Sentiment Analysis with BERT + LoRA")
    st.markdown("""
    This app analyzes the sentiment of your text and classifies it as **Negative**, **Neutral**, or **Positive**.
    The model is fine-tuned using LoRA (Low-Rank Adaptation) on BERT base.
    """)
    
    # Load model
    with st.spinner("Loading model... This might take a few seconds."):
        model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        st.error("Failed to load model. Please check if model files are available.")
        return
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter your text")
        text_input = st.text_area(
            "Type or paste your text here:",
            placeholder="I absolutely love this product! It's amazing...",
            height=150
        )
        
        predict_btn = st.button("Analyze Sentiment", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Quick Examples")
        examples = [
            "I love this product! It's amazing!",
            "It's okay, nothing special.",
            "This is the worst service ever.",
            "The food was average, not bad but not great.",
            "Best purchase I've made in years!"
        ]
        
        for example in examples:
            if st.button(example, use_container_width=True):
                st.session_state.example_text = example
    
    # Use example text if selected
    if 'example_text' in st.session_state:
        text_input = st.session_state.example_text
        del st.session_state.example_text
    
    # Prediction
    if predict_btn and text_input:
        with st.spinner("Analyzing sentiment..."):
            prediction, probabilities = predict_sentiment(text_input, model, tokenizer)
        
        if prediction is not None:
            st.subheader("Results")
            
            # Display prediction with emoji
            emoji_map = {"Negative": "😠", "Neutral": "😐", "Positive": "😊"}
            emoji = emoji_map.get(prediction, "🤔")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown(f"""
                <div style='text-align: center; padding: 20px; border-radius: 10px; background-color: #f0f2f6;'>
                    <h1 style='margin: 0;'>{emoji}</h1>
                    <h2 style='margin: 10px 0; color: {"#ff4b4b" if prediction == "Negative" else "#ffa500" if prediction == "Neutral" else "#00cc96"};'>{prediction}</h2>
                    <p style='margin: 0; font-size: 18px;'>Confidence: {probabilities.max():.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence bars
            st.subheader("Confidence Distribution")
            
            for i, (label, prob) in enumerate(zip(id2label.values(), probabilities)):
                color = "#ff4b4b" if label == "Negative" else "#ffa500" if label == "Neutral" else "#00cc96"
                bar_width = int(prob * 100)
                
                st.markdown(f"""
                <div style='margin: 10px 0;'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                        <span><b>{label}</b></span>
                        <span>{prob:.1%}</span>
                    </div>
                    <div style='background-color: #e0e0e0; border-radius: 10px; height: 20px;'>
                        <div style='background-color: {color}; width: {bar_width}%; height: 100%; border-radius: 10px;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Raw probabilities table
            with st.expander("View detailed probabilities"):
                prob_data = {
                    "Sentiment": list(id2label.values()),
                    "Probability": [f"{p:.3f}" for p in probabilities],
                    "Percentage": [f"{p:.1%}" for p in probabilities]
                }
                st.table(prob_data)

    elif predict_btn and not text_input:
        st.warning("Please enter some text to analyze.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit, Transformers, and PEFT (LoRA)</p>
        <p>Model: bert-base-uncased + LoRA fine-tuning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
