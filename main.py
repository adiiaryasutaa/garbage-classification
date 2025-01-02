import tensorflow as tf
import cv2
import numpy as np

# Load the model
model = tf.keras.models.load_model('garbage_classification_model (1).h5')

# Define the class names
class_names = ['battery', 'carton', 'metal', 'organic', 'paper', 'plastic', 'glass', 'clothing']

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to match the input size of the model
    img = cv2.resize(frame, (128, 128))
    #Normalize
    img = img / 255.0  
    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    confidence_percent = confidence * 100

    # Display the prediction on the frame
    cv2.putText(
        img=frame, 
        text=f"Prediction: {predicted_class} ({(confidence_percent):.2f}%)", 
        org=(10, 30), 
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=0.9, 
        color=(0, 255, 0), 
        thickness=2
    )

    cv2.imshow('Garbage Classification', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy windows
cap.release()
cv2.destroyAllWindows()