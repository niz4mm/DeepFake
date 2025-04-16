from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# EPOCHS = 10

# history = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=EPOCHS
# )
# Evaluate the model
# test_loss, test_acc = model.evaluate(val_generator)
# print(f"Validation Accuracy: {test_acc:.2f}")


model.save("ai_image_detector.h5")
model.save("C:/Users/PC/Desktop/project deepfake/model.h5")
