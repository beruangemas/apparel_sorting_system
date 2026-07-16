import gradio as gr
import numpy as np
from PIL import Image

# Import your custom modules
from data_handler import load_and_preprocess_data
from model_builder import create_cnn_model
from trainer import train_and_evaluate

def run_project():
    # Get the data and dynamically find the class names
    train_data, test_data, class_names = load_and_preprocess_data()

    # Build the model
    cnn_model = create_cnn_model(num_classes=len(class_names))

    # Train and evaluate the model
    trained_model = train_and_evaluate(cnn_model, train_data, test_data, class_names)

    # Define the GUI prediction logic
    def predict_item(uploaded_image):
        # Convert to RGB
        img = uploaded_image.convert('RGB')
        
        # Resize to our new HD standard
        img = img.resize((128, 128))
        
        # Normalize and reshape for the HD CNN (1 image, 128x128 pixels, 3 color channels)
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 128, 128, 3)

        # Predict using the trained model
        predictions = trained_model.predict(img_array)[0]
        
        # Dynamically loop through however many classes we have (e.g., 5)
        return {class_names[i]: float(predictions[i]) for i in range(len(class_names))}

    # 5. Set up and launch the GUI
    interface = gr.Interface(
        fn=predict_item,
        inputs=gr.Image(type="pil", label="Upload an Item (Jacket, Shirt, etc.)"),
        outputs=gr.Label(num_top_classes=3, label="AI Apparel Sorting Prediction"),
        title="📦 HD Apparel Sorter",
        description="Upload a full-color picture of an item. The upgraded HD AI will analyze it and sort it into the correct warehouse category.",
        #theme="default"
    )

    print("🌐 Launching Gradio Interface...")
    interface.launch(debug=True) 

if __name__ == "__main__":
    run_project()