# ResNet101-InceptionV3_Advanced-Traffic-Sign-Detection
Advanced Traffic Sign Detection Using a Hybrid Deep Learning Architecture with Enhanced Dense Layer Customization



This project demonstrates a hybrid deep learning approach for multi-class image classification using ResNet101 and InceptionV3. The model is trained on the ICTS Cropped Dataset and utilizes transfer learning and feature fusion to enhance classification performance.

### Project Overview
The research work includes data preprocessing and augmentation, balancing of training data through oversampling, transfer learning using ResNet101 and InceptionV3, fusion of features via the Keras functional API, model training with learning rate and early stopping callbacks, evaluation using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC curves.

### Dataset
The dataset consists of cropped images categorized into multiple classes for classification.
The data is loaded using CSV files pointing to image directories, and split into training, validation, and test sets. Data augmentation is applied using Keras’ ImageDataGenerator.
The dataset used in this project is the ICTS Cropped Dataset.

### Technologies Used
Python, TensorFlow / Keras, NumPy, Pandas, Matplotlib (for visualization), Scikit-learn (for evaluation metrics)

### Model Architecture
The hybrid model extracts features from ResNet101 and InceptionV3, merges them using the Keras functional API, and passes them through dense layers for classification.
Includes Convolutional base from pre-trained models, Feature fusion via concatenation, Dropout for regularization, Dense layers with softmax activation for final classification.
![model_methodology](https://github.com/user-attachments/assets/63ac34d9-7e63-41d4-a0cb-aaab49698aa1)


### Results and the Evaluation metrices
The proposed model achieved high accuracy and strong classification performance across all classes, Balanced dataset significantly improved model generalization, Evaluation metrics and ROC-AUC plots indicate robust predictive power and the confusion matrix and classification report confirm consistent results on test data.

### Accuracy Curve
![accuracy_curve](https://github.com/user-attachments/assets/e1a374d3-de8f-43ec-8e4e-b390c363bd05)

### Loss Curve
![loss_curve](https://github.com/user-attachments/assets/f7a043d4-fc30-45ca-8889-3325aad29ed9)

### ROC Curve
![roc_curve](https://github.com/user-attachments/assets/874bbd5d-d1a8-4a93-b602-1b545902469e)


### Confusion Matrix
![confusion_matrix](https://github.com/user-attachments/assets/c58c79b1-0b28-4dcf-8e6e-4cd1dc1ffd9b)



