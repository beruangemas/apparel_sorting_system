import tensorflow as tf
import numpy as np
import os

# We set a standard high-def size. 128x128 is perfect for your Mac M4 Pro.
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

def load_and_preprocess_data():
    # The folder where your extraction script saved the exact rubric amounts
    dataset_path = "hd_dataset"
    
    print("📂 Scanning local folders for Training Data...")
    # Load the images directly from your local folders
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(dataset_path, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int', 
        shuffle=True
    )

    print("\n📂 Scanning local folders for Testing Data...")
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(dataset_path, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int',
        shuffle=False
    )

    # Extract the exact 5 class names dynamically from the folder names
    class_names = train_dataset.class_names
    print(f"\n🎯 Found 5 categories: {class_names}")

    # Normalize the pixels from 0-255 to 0-1 (Still strictly required!)
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

    # Convert the modern TensorFlow Dataset objects back into standard Numpy Arrays.
    
    def dataset_to_numpy(dataset):
        images, labels = [], []
        for img_batch, label_batch in dataset:
            images.append(img_batch.numpy())
            labels.append(label_batch.numpy())
        return np.concatenate(images), np.concatenate(labels)

    print("⏳ Converting images to arrays for the AI to process...")
    train_images, train_labels = dataset_to_numpy(train_dataset)
    test_images, test_labels = dataset_to_numpy(test_dataset)

    print("✅ HD Dataset loaded, normalized, and converted successfully!")
    print(f"Total Training Images: {len(train_images)}")
    print(f"Total Testing Images: {len(test_images)}")
    
    return (train_images, train_labels), (test_images, test_labels), class_names