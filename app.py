import numpy as np
import tensorflow as tf
import cv2

# Load the TFLite model
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=r'C:\Users\lenovo\Desktop\Face-mask-detection-main\mask_detection.tflite')
    interpreter.allocate_tensors()
    return interpreter

# Preprocess the input image
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    img = cv2.resize(img, (96, 96)) 
    img = img.astype(np.float32) / 255.0 
    return np.expand_dims(img, axis=0) 

# Run inference on the image
def run_inference(interpreter, input_image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()

    # Get the output tensor and return the predicted class
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data, axis=1)
    return predicted_class[0]

def main():
    model_path = r'C:\Users\lenovo\Desktop\Face-mask-detection-main\mask_detection.tflite'  # Path to your TFLite model
    interpreter = load_model(model_path)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Preprocess the frame
        input_image = preprocess_image(frame)
        
        # Run inference
        predicted_class = run_inference(interpreter, input_image)
        
        # Display the result
        if predicted_class == 0:
            label = "No Mask"
            color = (0, 0, 255)  # Red
        else:
            label = "Mask"
            color = (0, 255, 0)  # Green

        # Display the label on the frame
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Show the frame with the prediction
        cv2.imshow("Mask Detection", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close any open windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
