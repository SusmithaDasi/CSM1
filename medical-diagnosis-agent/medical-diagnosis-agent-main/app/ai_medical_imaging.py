import streamlit as st
import os
import io
import base64
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import numpy as np
import cv2
import pydicom

# ───────────────────────────────────────────────
# Load API key from .env
# ───────────────────────────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ───────────────────────────────────────────────
# Streamlit Page Setup
# ───────────────────────────────────────────────
st.set_page_config(page_title="Medical Imaging Diagnosis Agent", layout="wide")
st.title("🩺 Medical Imaging Diagnosis Agent (GPT-4o Vision)")

st.markdown("""
Upload any **medical image** (X-ray, MRI, CT, or ultrasound) below.  
The AI will identify the image type, region, and possible abnormalities,
and explain its findings in both **technical** and **patient-friendly** terms.
""")

# ───────────────────────────────────────────────
# File Uploader
# ───────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a medical image", 
    type=["jpg", "jpeg", "png", "dcm"]
)

# ───────────────────────────────────────────────
# Helper: Convert uploaded file to PNG bytes
# ───────────────────────────────────────────────
def process_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith(".dcm"):
        ds = pydicom.dcmread(uploaded_file)
        img = ds.pixel_array
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        image_bytes = cv2.imencode(".png", img)[1].tobytes()
        return img, image_bytes, "DICOM"
    else:
        image = Image.open(uploaded_file).convert("RGB")
        image_bytes_io = io.BytesIO()
        image.save(image_bytes_io, format="PNG")
        return image, image_bytes_io.getvalue(), "Standard"

# ───────────────────────────────────────────────
# When an image is uploaded
# ───────────────────────────────────────────────
if uploaded_file:
    try:
        image_display, image_bytes, file_type = process_uploaded_file(uploaded_file)
        st.image(image_display, caption=f"Uploaded {file_type} image", use_column_width=True)
        st.info("Analyzing image using GPT-4o...")

        # Convert to base64 for OpenAI
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Prompt to the model
        prompt = """
        You are a board-certified radiologist and medical imaging expert.
        Analyze the uploaded medical image and provide:

        1. **Image Type** — X-ray, MRI, CT, or Ultrasound  
        2. **Anatomical Region** — what part of the body is visible  
        3. **Key Findings** — notable features, densities, or abnormalities  
        4. **Possible Diagnoses** — rank top differential diagnoses with reasoning  
        5. **Severity Assessment** — mild / moderate / severe  
        6. **Patient-Friendly Summary** — explain in everyday language  

        Format the response in **Markdown** for clarity.
        """

        # ───────────────────────────────────────────────
        # Send request to GPT-4o
        # ───────────────────────────────────────────────
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful medical imaging diagnosis assistant."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
        )

        # ───────────────────────────────────────────────
        # Display Results
        # ───────────────────────────────────────────────
        st.success("✅ Analysis Complete")
        st.markdown(result.choices[0].message.content)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

else:
    st.info("👆 Upload a medical image to begin analysis.")
