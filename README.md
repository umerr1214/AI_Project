
# 🍓 Fruit Image Classification Web App

This project is a complete pipeline for classifying images of fruits into one of **10 categories** using a **Convolutional Neural Network (CNN)** built with **PyTorch**. A simple **Flask-based web interface** is included to allow users to upload an image and receive the predicted fruit type.

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

---

## 🖥️ Web Interface

- Created using **Flask**
- Users can upload an image via browser
- The app predicts the fruit class and displays it

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
- Normalized using `[0.5, 0.5, 0.5]` mean and std
- CNN trained and saved using:

```python
torch.save(model.state_dict(), 'fruit_model.pth')
```

---

## 🖼️ Prediction Pipeline

1. User uploads image
2. Flask receives and preprocesses the image
3. Model predicts the class
4. Prediction is returned and shown on the web page

---

## ✅ Future Improvements

- Add confidence scores
- Use more complex models (e.g., ResNet)
- Deploy on Render, Hugging Face Spaces, or Streamlit
- Add image preview and drag-and-drop UI

---

## 📸 Screenshot

![Web App Preview](screenshot.png)

---

## 🔖 License

This project is for educational purposes and personal use.
