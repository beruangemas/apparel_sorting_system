import os
import zipfile
import shutil

# Authenticate with your copied Kaggle credentials
os.environ['KAGGLE_USERNAME'] = "your_kaggle_username"  # Replace with your actual Kaggle username
os.environ['KAGGLE_KEY'] = "your_kaggle_key"  # Replace with your actual Kaggle key

# Your specific dataset identifier
DATASET_IDENTIFIER = "gabrielalbertin/clothing-dataset"

def download_and_extract_exact_amount():
    print("📦 Downloading your specific Clothing Dataset from Kaggle...")
    os.system(f"kaggle datasets download {DATASET_IDENTIFIER}")
    
    zip_filename = "clothing-dataset.zip"
    
    if not os.path.exists(zip_filename):
        print("❌ ERROR: Download failed. Please check your credentials.")
        return

    print("📂 Analyzing the zip file and extracting the exact rubric amount...")
    
    # Set up our final folder structure
    base_dir = "hd_dataset"
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        all_files = zip_ref.namelist()
        
        # Group files by their class folder (e.g., 'jacket', 'shirt')
        class_files = {}
        for file_path in all_files:
            # Skip directories and non-images
            if file_path.endswith('/') or not file_path.lower().endswith(('.png', '.jpg', '.jpeg')): 
                continue
            
            # Find the class name from the folder path (e.g., "Dataset/jacket/img.jpg" -> "jacket")
            parts = file_path.split('/')
            if len(parts) >= 2:
                class_name = parts[-2] 
                if class_name not in class_files:
                    class_files[class_name] = []
                class_files[class_name].append(file_path)
        
        # Filter to find 5 classes that have at least 70 images to prevent crashing!
        valid_classes = {k: v for k, v in class_files.items() if len(v) >= 70}
        selected_classes = list(valid_classes.keys())[:5]
        
        if len(selected_classes) < 5:
            print("❌ ERROR: Could not find 5 classes with at least 70 images each.")
            return
            
        print(f"🎯 Selected Classes: {selected_classes}")

        # Extract exactly 50 for training and 20 for testing
        for class_name in selected_classes:
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
            
            files = valid_classes[class_name]
            
            # Extract Training Images (0 to 49)
            for i in range(50):
                source_path = files[i]
                target_path = os.path.join(train_dir, class_name, os.path.basename(source_path))
                with zip_ref.open(source_path) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                    
            # Extract Testing Images (50 to 69)
            for i in range(50, 70):
                source_path = files[i]
                target_path = os.path.join(test_dir, class_name, os.path.basename(source_path))
                with zip_ref.open(source_path) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)

    # Clean up the mess
    print("🧹 Cleaning up the massive zip file to save hard drive space...")
    os.remove(zip_filename)
    
    print("✅ SUCCESS: Exactly 50 Train and 20 Test images per class are beautifully organized in 'hd_dataset'!")

if __name__ == "__main__":
    download_and_extract_exact_amount()