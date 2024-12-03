from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(train_dir, val_dir, target_size=(150, 150), batch_size=20):
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    
    return train_generator, val_generator
