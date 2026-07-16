import tensorflow as tf

def create_cnn_model(num_classes=5):
    # 1. Build the High-Definition CNN Architecture
    model = tf.keras.models.Sequential([
        # Layer 1: Convolutional Layer (Updated for 128x128 RGB images)
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        
        # Layer 2: Pooling Layer (Compresses the 128x128 map)
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Layer 3: SECOND Convolutional Layer (Crucial for high-def details!)
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Layer 4: SECOND Pooling Layer (Compresses the map again before flattening)
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Layer 5: Flattening Layer (Turns the 2D maps into a 1D list)
        tf.keras.layers.Flatten(),
        
        # Layer 6: Dense Hidden Layer (The core learning brain)
        tf.keras.layers.Dense(128, activation='relu'),
        
        # Layer 7: Output Layer (Dynamic nodes for our specific classes)
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # 2. Compile the Model
    # We keep 'sparse_categorical_crossentropy' because our data_handler outputs integer labels (0, 1, 2, 3, 4)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("✅ HD CNN Architecture successfully built!")
    model.summary()
    return model