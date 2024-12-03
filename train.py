import os
from src.data_loader import load_data
from src.model import create_model, train_model

# Путь к директориям с данными
train_dir = 'data/train'
val_dir = 'data/validation'

# Загрузка данных
train_data, val_data = load_data(train_dir, val_dir)

# Создание модели
model = create_model()

# Обучение модели
history = train_model(model, train_data, val_data)

# Сохранение модели
model.save('models/car_detector.h5')
