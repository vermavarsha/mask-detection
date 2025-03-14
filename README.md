# mask-detection

This project focuses on detecting whether a person is wearing a mask or not using deep learning. The model was trained on a dataset of masked and unmasked faces and later converted to TensorFlow Lite (TFLite) for lightweight deployment.

Dataset Details
Dataset Used: A collection of images containing individuals with and without masks.

Preprocessing Steps:
Resized images to 96×96 pixels for faster computation.
Normalized pixel values to the range [0,1].
Augmented images (rotation, flipping) to improve generalization.

Model Architecture

Type: Convolutional Neural Network (CNN)
Input Shape: (96, 96, 3) – RGB images

Layers:
Convolutional layers with ReLU activation
Batch Normalization and MaxPooling
Fully Connected Dense layers
Softmax output for binary classification (Mask/No Mask)

Training Process
Loss Function: Sparse Categorical Crossentropy
Optimizer: Adam (learning rate = 0.001)
Metrics: Accuracy

Techniques Used:
Early stopping to prevent overfitting
Learning rate reduction on plateau
Class weighting for handling imbalance

Challenges Faced & Solutions
Class Imbalance: Resolved using class weighting and data augmentation.
Overfitting: Prevented using early stopping and dropout layers.
Performance on Edge Devices: Optimized the model size by converting to TFLite.

Conclusion
The project successfully implements a real-time mask detection system using deep learning. The trained model, after conversion to TFLite, is lightweight and can be deployed efficiently on various platforms, making it suitable for real-world applications in public safety and healthcare monitoring.
