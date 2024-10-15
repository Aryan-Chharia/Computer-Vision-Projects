import requests
from PIL import Image
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Initialize processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda" if torch.cuda.is_available() else "cpu")

# Function to process and caption an image from a URL
def caption_image(image_url):
    try:
        # Load image from the provided URL
        raw_image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
        
        # Conditional image captioning
        text = "a photography of"
        inputs = processor(raw_image, text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        out = model.generate(**inputs)
        conditional_caption = processor.decode(out[0], skip_special_tokens=True)

        # Unconditional image captioning
        inputs = processor(raw_image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        out = model.generate(**inputs)
        unconditional_caption = processor.decode(out[0], skip_special_tokens=True)

        return raw_image, conditional_caption, unconditional_caption
    
    except Exception as e:
        st.error(f"Error occurred: {e}")
        return None, None, None

# Streamlit App
st.title("Image captioning model")

# Input field for image URL
image_url = st.text_input("Enter the image URL:", "")


# Process and display captions when the user submits an image URL
if st.button("Generate Captions"):
    if image_url:
        with st.spinner("Processing..."):
            raw_image, conditional_caption, unconditional_caption = caption_image(image_url)
        
        if raw_image:
            # Display the image
            st.image(raw_image, caption="Uploaded Image", use_column_width=True)
            
            # Display captions
            st.subheader("Generated Captions:")
            st.write(f"**Conditional Caption:** {conditional_caption}")
            st.write(f"**Unconditional Caption:** {unconditional_caption}")
    else:
        st.error("Please enter a valid image URL.")
