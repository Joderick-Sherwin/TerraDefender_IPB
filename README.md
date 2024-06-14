# TerraDefender IPB System

## Overview

The TerraDefender IPB (Intelligence Preparation of the Battlefield) System is an advanced terrain analysis tool designed to support military and defense operations. By leveraging state-of-the-art convolutional neural networks (CNNs), the system can accurately classify various terrain types, providing critical insights for strategic planning and situational awareness.

## Project Demo

Watch the demo video on [Google Drive](https://drive.google.com/file/d/111O5vvSD_WlYcgX0USCwvWx9OXTtowci/view?usp=sharing).

<div align="center">
  <a href="https://drive.google.com/file/d/111O5vvSD_WlYcgX0USCwvWx9OXTtowci/view?usp=sharing">
    <img src="https://github.com/Joderick-Sherwin/TerraDefender_IPB/blob/main/TerraDefender_Loading_Page.jpeg" alt="Watch the video" style="width:400px;"/>
  </a>
</div>

- **Input**
![alt text](https://github.com/Joderick-Sherwin/TerraDefender_IPB/blob/main/Sample%20Input.jpg)

- **Output**
![alt text](https://github.com/Joderick-Sherwin/TerraDefender_IPB/blob/main/Sample%20Output.jpg)

## Features

- **Terrain Classification**: The system is capable of identifying and classifying multiple terrain types, including grassy, marshy, sandy, and rocky areas. This classification is essential for understanding the operational environment and making informed decisions in military contexts.
- **Data Augmentation**: To improve model robustness and generalization, the system employs various data augmentation techniques. These include rotation, shifting, shearing, zooming, and horizontal flipping of images. These augmentations help the model learn to recognize terrain features from different perspectives and under varied conditions.
- **Regularization**: The system incorporates dropout regularization to prevent overfitting and ensure the model generalizes well to new, unseen data. Dropout is applied to the fully connected layers during training, randomly setting a fraction of input units to zero at each update to reduce dependency on specific neurons.
- **Performance Metrics**: Comprehensive evaluation metrics are used to assess the model's performance. These include accuracy, precision, recall, and F1-score, which together provide a detailed understanding of the model's effectiveness in classifying terrain types.

## Data Augmentation

Data augmentation techniques are crucial for enhancing the robustness of the model. By artificially expanding the training dataset, these techniques help the model generalize better to new data:

- **Rotation**: Randomly rotates images up to 20 degrees, helping the model learn to recognize features from different angles.
- **Width and Height Shift**: Randomly shifts images horizontally or vertically by up to 20% of the image dimensions, aiding the model in learning positional invariance.
- **Shear**: Applies random shearing transformations, which helps the model become invariant to affine transformations.
- **Zoom**: Randomly zooms in or out on images, allowing the model to recognize features at different scales.
- **Horizontal Flip**: Randomly flips images horizontally, which is useful for learning symmetry in terrain features.
- **Fill Mode**: Uses nearest neighbor filling to handle any new pixels introduced during augmentation.

## Training and Evaluation

### Learning Rate Schedule

The learning rate is a critical hyperparameter that controls how much the model's weights are adjusted during training. The TerraDefender IPB System uses a dynamic learning rate schedule to optimize training:

- **Epochs 0-2**: Learning rate = 1e-4
- **Epochs 3-5**: Learning rate = 1e-5
- **Epochs 6+**: Learning rate = 1e-6

### Callbacks

Several callbacks are employed during training to enhance model performance and prevent overfitting:

- **Early Stopping**: Monitors validation loss and stops training if no improvement is observed for 10 epochs. This helps in preventing overfitting and saves computational resources.
- **Model Checkpoint**: Saves the best model based on validation loss, ensuring that the most performant model is retained.
- **Learning Rate Scheduler**: Adjusts the learning rate according to the predefined schedule to ensure smooth and efficient convergence during training.

### Evaluation Metrics

The system's performance is evaluated using a comprehensive set of metrics:

- **Accuracy**: Measures the proportion of correctly classified samples.
- **Precision**: The ratio of true positive predictions to the total predicted positives, indicating the accuracy of the positive predictions.
- **Recall**: The ratio of true positive predictions to the total actual positives, reflecting the model's ability to capture all positive samples.
- **F1-Score**: The harmonic mean of precision and recall, providing a balance between the two metrics.

## Results

Upon training and evaluation, the TerraDefender IPB System generates detailed classification reports and confusion matrices. These outputs provide insights into the model's accuracy, error distribution, and areas for improvement. By analyzing these results, users can understand how well the model performs across different terrain types and make informed adjustments to enhance its accuracy.

<table>
  <tr>
    <td>
      <img src="https://github.com/Joderick-Sherwin/TerraDefender_IPB/blob/main/Figure_1.png" alt="Figure 1" width="400"/>
    </td>
    <td>
      <img src="https://github.com/Joderick-Sherwin/TerraDefender_IPB/blob/main/Figure_2.png" alt="Figure 2" width="400"/>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/Joderick-Sherwin/TerraDefender_IPB/blob/main/Figure_3.png" alt="Figure 3" width="400"/>
    </td>
    <td>
      <img src="https://github.com/Joderick-Sherwin/TerraDefender_IPB/blob/main/Figure_4.png" alt="Figure 4" width="400"/>
    </td>
  </tr>
</table>

## Future Work

Future improvements for the TerraDefender IPB System may include:

- **Enhanced Data Augmentation**: Incorporating more advanced augmentation techniques, such as synthetic data generation or domain adaptation, to further improve model robustness.
- **Model Architecture Optimization**: Experimenting with deeper or more complex CNN architectures, such as ResNet or EfficientNet, to enhance feature extraction capabilities.
- **Transfer Learning**: Leveraging pre-trained models on large-scale datasets to improve performance with limited training data.
- **Real-Time Analysis**: Developing capabilities for real-time terrain analysis using edge computing or optimized inference engines.
- **Expanded Terrain Classes**: Increasing the number of terrain classes to cover a broader range of environments and operational scenarios.

## Acknowledgements

We acknowledge the contributions of the open-source community and the use of TensorFlow and Keras libraries in developing this system. Their extensive documentation and community support have been invaluable in building and refining the TerraDefender IPB System.

---

For more information, visit our [project page](https://github.com/your-repo/TerraDefender).
