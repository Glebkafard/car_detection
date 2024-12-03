import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

def load_image(image_path, target_size=(150, 150)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(model, image_path):
    img_array = load_image(image_path)
    prediction = model.predict(img_array)
    if prediction > 0.5:
        return "На изображении есть автомобиль"
    else:
        return "Автомобиля нет"

def load_trained_model(model_path='models/car_detector.h5'):
    return load_model(model_path)
