
# 🍓 Fruit Image Classification Web App

This project is a complete pipeline for classifying images of fruits into one of **10 categories** using a **Convolutional Neural Network (CNN)** built with **PyTorch**.It includes a visually appealing **Flask-based web interface** where users can upload an image and receive the predicted fruit type.

---

## 📁 Dataset

- **Source**: [Kaggle - Fruit Classification (10-class)](https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class)
- The dataset consists of images categorized into 10 fruit classes such as:
  - Apple, Banana, Blueberry, Guava, Mango, Orange, Peach, Pineapple, Strawberry, Watermelon

---

## 🧠 Model

- A **simple CNN** with:
  - 2 convolutional layers
  - 2 max pooling layers
  - 2 fully connected layers
- Built using `torch.nn.Module`
- Trained on the fruit dataset using basic transformations and normalization
- **Not being used now**

---

### 🔧 Architecture Details:

We've upgraded the model from a basic CNN to a more powerful custom-designed **ImprovedCNN** architecture.

- **3 Convolutional Layers** with increasing filters: 32 → 64 → 128
- Each followed by **Batch Normalization**, **ReLU**, and **MaxPooling**
- **Dropout layers** added for regularization to prevent overfitting
- Two **Fully Connected (Dense)** layers:
  - First FC layer: `128 * 16 * 16 → 256`
  - Output FC layer: `256 → 10 (classes)`
- Trained with **CrossEntropyLoss** and **Adam optimizer**

This architecture achieves significantly better accuracy and generalization on the test set compared to the original model.

---

## 🖥️ Web Interface

- Created using **Flask**
- Users can upload an image from the browser
- Predicted fruit class is displayed after image upload

---

## 📂 Project Structure

```
fruit_classifier/
├── app.py                 # Flask app to serve the model
├── model.py               # CNN model definition
├── fruit_model.pth        # Trained PyTorch model weights
├── templates/
│   └── index.html         # HTML interface for file upload and prediction
```

---

## 🚀 How to Run

### 1. Install Requirements

```bash
pip install flask torch torchvision pillow
```

### 2. Run the Flask App

```bash
python app.py
```

Then open your browser at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🏗️ Model Training Summary

- Images resized to 128x128
- Normalized using [0.5, 0.5, 0.5] for mean and std
- Trained with:
- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Epochs: typically 10–20 for convergence

- Model saved via:

```python
torch.save(model.state_dict(), 'fruit_model.pth')
```

---

## 🖼️ Prediction Pipeline

1. User uploads image via browser
2. Flask handles and transforms the image
3. ImprovedCNN predicts the fruit class
4. Prediction displayed dynamically on the web page

---

## 🔖 License

This project is for educational purposes and personal use.
