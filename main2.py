from fastapi import FastAPI, File, UploadFile
import uvicorn
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model("a_model.h5")

CLASS_NAMES = ['Mild_Demented','Moderate_Demented','non_Demented','Very_Mild_Demented']

# Define input image size
IMG_HEIGHT = 128
IMG_WIDTH = 128

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

def resize_image(image, size=(128, 128)):
    """Resize the image to the given size"""
    img = Image.fromarray(image)
    img = img.resize(size)
    return np.array(img)

# Define preprocessing function
# def preprocess_image(image_bytes):
    # img = Image.open(BytesIO(image_bytes)).convert('RGB')
    # img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    # img_array = np.asarray(img)
    # img_array_copy = np.copy(img_array)
    # img_array_float = img_array.astype('float64')
    # img_array_float /= 255.
    # img_array_expanded = np.expand_dims(img_array_float, axis=0)
    # return img_array_expanded

# Define prediction function
# @app.post("/predict")
# async def predict_alzheimer(image: bytes = File(...)):
#     img = preprocess_image(image)
#     predictions = MODEL.predict(img)
#     predicted_class_index = np.argmax(predictions, axis=-1)[0]
#     predicted_class_name = CLASS_NAMES[predicted_class_index]
#     return {"prediction": predicted_class_name}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image = resize_image(image)
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)

    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class_name,
        'confidence': float(confidence)
    }

# @app.post("/predict")
# async def predict(
#     file: UploadFile = File(...)
# ):
#     image = read_file_as_image(await file.read())
#     image = resize_image(image)
#     img_batch = np.expand_dims(image, 0)
    
#     predictions = MODEL.predict(img_batch)

#     predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#     confidence = np.max(predictions[0])
#     return {
#         'class': predicted_class,
#         'confidence': float(confidence)
#     }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)