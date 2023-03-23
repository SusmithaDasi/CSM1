import io
import cv2
import numpy as np
import pydicom
from PIL import Image

def process_uploaded_file(uploaded_file):
    """
    Detects if the uploaded file is a DICOM or a standard image,
    and converts it to bytes usable by the OpenAI API.
    """
    if uploaded_file.name.endswith(".dcm"):
        ds = pydicom.dcmread(uploaded_file)
        img = ds.pixel_array
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        image_bytes = cv2.imencode(".png", img)[1].tobytes()
        return img, image_bytes, "DICOM"
    else:
        image = Image.open(uploaded_file)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        return image, image_bytes.getvalue(), "Standard"
