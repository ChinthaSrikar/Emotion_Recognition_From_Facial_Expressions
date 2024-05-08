
# Emotion Recognition from Facial Expressions

### Abstract
This project presents a novel approach to Facial Emotion Recognition (FER) using a custom-designed Five-CNN architecture. By meticulously optimizing hyperparameters and exploring diverse optimization techniques, we achieved a single-network accuracy of 64.10% on the FER2013 dataset, setting a new benchmark for this task.

### Dataset: FER2013
The FER2013 dataset consists of over 35,000 images labeled with seven different emotions. These images are standardized to 48x48 pixels and converted to grayscale to simplify processing. Data augmentation techniques such as horizontal flipping and random rotation are used to enhance the training data.

### Model Architecture: Five-CNN
Our model features an input layer designed for 48 *48 pixel grayscale images resized to 98*98 pixel, followed by five convolutional blocks, adaptive average pooling, and several fully connected layers. This architecture effectively captures a range of features from simple edges to complex objects, making it highly suitable for emotion recognition.

### Loss Function: Cross-Entropy Loss
Cross-entropy loss was selected due to its effectiveness in handling the multi-class classification of emotions. It excels in measuring the discrepancy between predicted and actual probability distributions.

### Optimizer: Adam
The Adam optimizer was chosen for its adaptive learning rate capabilities and momentum integration, which enhance the training stability and efficiency.

### Results
Our model demonstrated robust performance, particularly in distinguishing various facial expressions under different lighting conditions and across diverse facial features.

### Conclusion
The project underscores significant advancements in the domain of emotion recognition, paving the way for enhanced applications in areas such as human-computer interaction, marketing, and healthcare.
