# Emotion-Detection-Model-CNN

![image_alt](https://imgs.search.brave.com/n-GzNMltNlKUS2xmXtBcP-AbarfpsAnnrkxHAzth5XM/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly93d3cu/em9ua2FmZWVkYmFj/ay5jb20vaHViZnMv/ZW1vdGlvbiUyMGRl/dGVjdGlvbi5wbmc)


## 🔭 Overview
This project develops a facial emotion recognition system using `Convolutional Neural Networks (CNNs)` applied to the `FER-2013` dataset. The system classifies facial expressions into seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The notebook implements and compares several model architectures, including custom CNNs and transfer learning approaches with pre-trained networks. This is fundamentally an image classification task, which falls under the broader domain of computer vision, even if specific libraries like OpenCV were not explicitly used—instead, the focus is on deep learning frameworks for processing and analyzing images.


### 🚀 Key topics covered (based on code analysis):
- **Image Processing and Deep Learning:** Handling image data, building CNN models, training, and evaluation using Keras/TensorFlow.
- **Transfer Learning:** Adapting pre-trained models (`ResNet50` and `VGG16`) from ImageNet for emotion detection.
- **Data Augmentation:** Techniques to enhance training data and improve model robustness.
- **Model Evaluation:** Logging training metrics, accuracy assessment, and potential visualizations.

The project runs in a Kaggle Jupyter Notebook with Python 3.12, leveraging libraries such as NumPy, Pandas, TensorFlow/Keras, and possibly Matplotlib for plots.

---

## 📁 Dataset

- **FER-2013:** Grayscale 48x48 pixel images of faces.
  - Training: Approximately `28,709` images.
  - PublicTest (Validation): `~3,589` images.
  - PrivateTest (Test): `~3,589` images.
  - Classes: 7 emotions as mentioned.
  - Loaded from: `/kaggle/input/fer2013/`, with images in subfolders by emotion (e.g., surprise images listed in code).

- Key Challenges: Class imbalance (e.g., fewer "disgust" samples), low-resolution images, and variations in poses/lighting.
- Code Verification: The first cell uses os.walk to list dataset files, confirming image paths for loading. Data is preprocessed into arrays, normalized (0-1 range), and labels are one-hot encoded.

---

## 🤖 Models Implemented
Four model variants are trained and saved in /kaggle/working/Emotion_detection_model(CNN)/. Each includes training logs (.log files) showing epoch-wise loss, accuracy, and validation metrics. Models use Adam optimizer, categorical crossentropy loss, and callbacks like EarlyStopping.

1. **Custom CNN (Baseline):**
    - Architecture: Sequential model with Conv2D layers (e.g., filters=32/64/128), MaxPooling2D, Dropout (for regularization), Flatten, and Dense layers leading to a 7-unit softmax output.
    - Input Shape: 48x48x1 (grayscale).
    - Training: Typically 20-50 epochs; accuracy ~50-60% based on logs.
    - Saved: Custom_CNN_Main/Custom_CNN_model.keras.
    - Includes: Architecture diagram (Architecture.png).
    - Code Verification: Cells define the model with Keras layers, compile, and fit on dataset generators.

2. **Custom CNN with Data Augmentation:**
    - Extends baseline using ImageDataGenerator for real-time augmentations (rotation, zoom, horizontal flip) to address overfitting and imbalance.
    - Saved: Custom_CNN_with_Augmentation/Custom_CNN_augmented_model.keras.
    - Code Verification: Generator configured with parameters like rotation_range=20, zoom_range=0.15; improves validation accuracy slightly over baseline.

3. **ResNet50 Transfer Learning:**
    - Base: Pre-trained ResNet50 (ImageNet weights), frozen layers, with custom top (GlobalAveragePooling2D, Dense).
    - Adaptation: Grayscale input handled by repeating channels or preprocessing; some layers unfrozen for fine-tuning.
    - Performance: Often ~65% accuracy, better than custom due to pre-learned features.
    - Saved: ResNet50_Transfer_Learning/ResNet50_Transfer_Learning.keras.
    - Code Verification: Imports from tensorflow.keras.applications import ResNet50; model built by adding layers atop base.
  
![image_alt](https://imgs.search.brave.com/hPXE1pkGJsps79GKbZKRgjwke8qv2lPpCRUZVnNdYVM/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly90b3dh/cmRzZGF0YXNjaWVu/Y2UuY29tL3dwLWNv/bnRlbnQvdXBsb2Fk/cy8yMDIyLzA4LzB0/SDlldnVPRnFrOEY0/MUZHLnBuZw)

4. **VGG16 Transfer Learning:**
    - Similar to ResNet50, using pre-trained VGG16 with custom classification head.
    - Saved: VGG16_Transfer_Learning/VGG16_Transfer_Learning.keras.
    - Code Verification: Imports from tensorflow.keras.applications import VGG16; similar structure, adapted for grayscale.
  
![image alt](https://imgs.search.brave.com/9sBV9D92O-LqsOCO8Xo-vKUWPNg4JQfJs4iBegmSqxk/rs:fit:0:180:1:0/g:ce/aHR0cHM6Ly9zdG9y/YWdlLmdvb2dsZWFw/aXMuY29tL2xkcy1t/ZWRpYS9pbWFnZXMv/dmdnMTYtYXJjaGl0/ZWN0dXJlLndpZHRo/LTEyMDAuanBn)



Training logs indicate progressive learning, with transfer models showing lower initial loss due to pre-training.
