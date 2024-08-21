# Action Recognition for Sign Language

## Overview
This project is focused on Action Recognition for Sign Language, which aims to bridge the communication gap between sign language users and those who may not understand sign language. The system leverages MediaPipe Holistic to track and analyze keypoints from the body, hands, and face, which are critical for understanding sign language gestures. These keypoints are then processed by a Neural Network to classify the actions being performed.

The primary goal of this project is to build a model that can accurately recognize specific sign language gestures and translate them into meaningful text. The current version of the model focuses on recognizing three key gestures: "Hello", "Thanks", and "I Love U". These gestures were chosen as they are commonly used in everyday communication.

## Features
- **Real-Time Sign Language Recognition**: The system recognizes and classifies various sign language gestures in real time.
- **MediaPipe Holistic**: Utilizes the powerful MediaPipe Holistic model to detect and track hand and body landmarks.
- **Simple Neural Network**: A custom neural network architecture is used to classify the actions based on the keypoints extracted from MediaPipe.
- **Accuracy**: Achieved an accuracy of **0.9882352941176471** on the validation dataset.
- **Predictions**: The model can predict actions such as "Hello", "Thanks", "I Love U" with reasonable accuracy.

## Dataset
The dataset consists of 30 data frames for each of the sign language actions, split into 3 segments with a total of 1662 keypoints. The data was preprocessed and used for training the neural network.

## Model Architecture
The neural network model consists of the following layers:
- **LSTM Layers**: 3 LSTM layers with 64, 128, and 64 units respectively.
- **Dense Layers**: 2 dense layers with 64 and 32 units.
- **Output Layer**: A softmax output layer that predicts the action class.

### Summary of the Model
- **Input Shape**: `(20, 1662)` representing 20 frames of data with 1662 keypoints per frame.
- **Training**: The model was trained for 2000 epochs with early stopping and tensorboard callbacks.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Asteramartya12/Sign-Language-to-voice-translator
   ```
2. Navigate to the project directory:
   ```bash
   cd Action-recognition-for-sign-lanaguage
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the project:
   ```bash
   jupyter notebook ActionRecognition.ipynb
   ```

### Steps Overview:
1. **Import Necessary Libraries:**

2. **Install and import essential libraries like TensorFlow, OpenCV, Mediapipe, NumPy, Matplotlib, and other required tools. These libraries help in building the action recognition model.
Initialize Mediapipe Holistic Model:**

3. **Use Mediapipe Holistic to detect keypoints (face, pose, left and right hand landmarks) from video frames captured via webcam. Mediapipe is a powerful tool for real-time keypoint detection.
Define Utility Functions:**

4. **mediapipe_detection(image, model): Converts images to RGB, processes them using the Mediapipe model, and converts them back to BGR.
draw_landmarks(image, results): Draws landmarks for face, pose, and hands on the image.
draw_styled_landmarks(image, results): Adds custom styles to the drawn landmarks for better visualization.
Capture Video Feed:**

5. **Capture video frames from the webcam and process them using Mediapipe to detect keypoints in real-time.
& Save Keypoints:**
![image](https://github.com/user-attachments/assets/55ca237a-a677-46b0-ad71-20db28770aa7)


7. **For each frame in the captured video, keypoints are extracted and saved as .npy files. These files contain the keypoint data needed for training the model.
Set Up Data Collection:**

8. **Create directories for storing keypoint data of different actions. This ensures organized storage of the collected data.
Collect Keypoint Data:**

9. **For each action (e.g., "Hello", "Thanks", "I Love U"), capture 30 sequences, each consisting of 30 frames. Keypoints are extracted and stored in the respective directories.
Preprocess Data:**

10. **Convert the stored keypoint data into a format suitable for training the model. The keypoint sequences are converted into NumPy arrays, and corresponding labels are created.
Train-Test Split:**

11. **Split the data into training and testing sets. This ensures that the model can be evaluated on unseen data.
Build LSTM Neural Network:**

12. **Define a Sequential LSTM model with multiple layers. The model is designed to recognize patterns in the sequences of keypoints.
Add Regularization Techniques:**

13. **Dropout layers and L2 regularization are added to prevent overfitting and improve the model's generalization capability.
Compile the Model:**

14. **The model is compiled with an Adam optimizer, categorical cross-entropy loss function, and accuracy as the evaluation metric.
Train the Model:**

15. **Train the model using the training data with early stopping and TensorBoard callback for monitoring. Early stopping helps prevent overfitting by stopping the training when validation loss stops improving.
Evaluate Model:**

16. **Evaluate the model's performance using a confusion matrix and accuracy score. This provides insights into how well the model is performing on the training data.
Test in Real-Time:**

**Model Summary**

![image](https://github.com/user-attachments/assets/c41e5c87-3573-4ee9-8e99-02b293271758)

**Accuracy Score**

![image](https://github.com/user-attachments/assets/e1e8e533-1d1a-42b0-928b-6a41d23087b8)


18. **Finally, the trained model is tested in real-time using the webcam feed. The model predicts the action being performed, and the results are visualized with confidence scores.
Visualization:**
![image](https://github.com/user-attachments/assets/d1d956dc-9313-4256-b060-d4fb576f2356)


The real-time predictions are visualized on the screen, along with the probability of each action, allowing for easy interpretation of the results.
## Usage
- **Real-time Prediction**: The system can be used for real-time prediction of sign language actions.
- **Pre-trained Model**: You can load the pre-trained model and test it on your own data, or you can run it on the pre-stored data.

## Results
- **Accuracy**: The model achieved an accuracy of **0.9882352941176471** on the validation set.
- **Example Predictions**:
  - Input: "Hello" gesture → Prediction: "Hello"
  - Input: "Thanks" gesture → Prediction: "Thanks"
  - Input: "I Love U" gesture → Prediction: "I Love U"

## Future Work
- **Expand the Dataset**: Add more sign language actions to improve the robustness of the model.
- **Optimize the Model**: Experiment with different neural network architectures and hyperparameters to enhance accuracy.
- **Deploy the Model**: Integrate the model into a web or mobile application for easy accessibility.

## Conclusion
This project demonstrates a basic but functional approach to recognizing sign language actions using MediaPipe Holistic and a simple neural network. The model shows promising results, and with further improvements, it could be a useful tool for sign language translation.

## Acknowledgements
- **MediaPipe**: For providing an excellent framework for real-time holistic tracking.
- **Keras/TensorFlow**: For building and training the neural network.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
