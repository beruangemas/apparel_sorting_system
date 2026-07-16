import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def train_and_evaluate(model, train_data, test_data, class_names):
    train_images, train_labels = train_data
    test_images, test_labels = test_data

    # Train the Model
    print("🚀 Starting the training process. Please wait...")
    history = model.fit(train_images, train_labels,
                        epochs=10,
                        batch_size=32,
                        validation_split=0.2)
    print("✅ Training Complete!")

    # Plot Training History
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show(block=False) # block=False allows the code to continue running

    # Test on Unseen Data
    print("🧠 Testing the AI on unseen warehouse items...")
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"\n✅ Final Test Accuracy: {test_acc*100:.2f}%")

    #Generate Predictions
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)

    #Print Classification Report
    print("\n📊 --- CLASSIFICATION REPORT ---")
    print(classification_report(test_labels, predicted_labels, target_names=class_names))

    #Generate and Plot Confusion Matrix
    print("\n🟩 --- CONFUSION MATRIX ---")
    cm = confusion_matrix(test_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Apparel Sorting Confusion Matrix')
    plt.ylabel('Actual True Item')
    plt.xlabel('AI Predicted Item')
    
    #Save the plotted confusion matrix to PNG for report
    print("📸 Saving Confusion Matrix to 'confusion_matrix.png' for your report...")
    plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=300)
    plt.close() # Safely closes the background drawing to free up Mac memory
    # --------------------------------------------------------------

    return model