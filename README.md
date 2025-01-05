Name: PODUGU RISHITHA

Company: CODTECH IT SOLUTIONS

ID: CT08DFJ

Domain: Machine Learning

Duration: December 2024 to January 2025

Mentor: NEELA SANTHOSH KUMAR


## **Convolutional Neural Network for Image Classification**

This repository contains a Python implementation of a Convolutional Neural Network (CNN) for image classification using the TensorFlow library. The model is trained and evaluated on the CIFAR-10 dataset, a standard dataset for image recognition tasks.

---

**Output**

![Screenshot 2025-01-05 12 34 33](https://github.com/user-attachments/assets/9a36f254-e111-41f3-b491-0d2e9949625c)
![image](https://github.com/user-attachments/assets/0d2df623-bdbf-4eea-9cc4-7cf64c0a2427)


### **Overview**

The code demonstrates:
1. Loading and preprocessing the CIFAR-10 dataset.
2. Building a CNN with TensorFlow and Keras.
3. Training the model to classify images into 10 categories.
4. Evaluating the model's performance on test data.

---

### **Steps in the Code**

1. **Import Libraries**  
   Essential libraries like TensorFlow and Keras are imported to build and train the CNN.

2. **Load and Preprocess Data**  
   - The CIFAR-10 dataset is loaded using `cifar10.load_data()`.
   - Data normalization is performed by scaling pixel values to the range [0, 1].

3. **Build CNN Model**  
   The CNN architecture includes:
   - **Convolutional Layers:** Extract spatial features from images.
   - **Pooling Layers:** Reduce spatial dimensions to minimize computation and overfitting.
   - **Fully Connected Layers:** Perform classification on the extracted features.
   - **Output Layer:** Uses a softmax activation function to classify images into 10 categories.

4. **Compile and Train the Model**  
   - **Loss Function:** `sparse_categorical_crossentropy` for multi-class classification.
   - **Optimizer:** Adam optimizer for efficient training.
   - **Metrics:** Accuracy to monitor performance.
   - The model is trained for 10 epochs, and validation is performed on test data.

5. **Evaluate the Model**  
   The model's performance is validated using the test set, and accuracy and loss are reported.

---

### **Requirements**

- Python 3.7 or higher
- TensorFlow 2.x
- NumPy

Install dependencies using:
```bash
pip install tensorflow numpy
```

---

### **Usage**

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cnn-image-classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd cnn-image-classification
   ```
3. Run the script:
   ```bash
   python cnn_image_classification.py
   ```

---

### **Results**

The model achieves approximately **70% accuracy** on the CIFAR-10 test set after 10 epochs of training. This accuracy can be improved by:
- Adding more layers or filters.
- Using data augmentation techniques.
- Increasing the number of epochs.

---

### **Dataset**

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. It is a commonly used dataset for benchmarking image classification models.

---

### **Contributing**

Feel free to contribute by:
- Improving the model architecture.
- Adding data augmentation for better generalization.
- Extending the code for other datasets.
  
