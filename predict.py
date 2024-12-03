import sys
from src.predict import predict_image, load_trained_model

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python predict.py <C:/Users/Gleb/Desktop/britania_test.jpg>")
        sys.exit(1)

    image_path = sys.argv[1]
    model = load_trained_model()
    result = predict_image(model, image_path)
    print(result)
