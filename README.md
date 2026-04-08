# 🩺 AI-Based Multi-class Chest X-ray Classification (COVID-19, Pneumonia, Normal)

This project implements a deep learning pipeline to classify chest X-ray images into:

- 🦠 COVID-19  
- 🫁 Viral Pneumonia  
- 🟢 Normal  

It also integrates **Grad-CAM explainability** and a **Streamlit web application** for interactive predictions.

---

## 🚀 Key Features

- ✅ Multi-class classification (3 classes)
- ✅ Custom CNN model (best performing)
- ✅ ResNet model (transfer learning comparison)
- 🔥 Grad-CAM for model interpretability
- 📊 Confidence score & probability visualization
- 🌐 Streamlit web app for real-time predictions
- 📦 Modular pipeline (preprocessing → training → deployment)

---

## 🧠 Problem Statement

Detect and differentiate between COVID-19, Pneumonia, and Normal chest X-rays using deep learning while ensuring model interpretability.

---

## 📂 Dataset

Due to large size, the dataset is not included in this repository.

👉 Download from Kaggle:  
https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset

---

## 📁 Project Structure
Multiclass_Image_Classification/
│
├── app.py # Streamlit app
├── requirements.txt
├── README.md
│
├── artifacts/ # ❗ Not included (large files)
│ ├── cnn_model.pth
│ ├── resnet_model.pth
│ ├── data_split.csv
│ ├── class_mapping.json
│
├── notebooks/
│ ├── preprocessing.ipynb
│ ├── eda.ipynb
│ ├── cnn_model.ipynb
│ ├── resnet_model.ipynb


---

## ⚠️ Important: Large Files Not Included

The following are excluded from GitHub due to size:

- Trained model weights (`.pth`)
- Dataset files
- Processed splits
- Class mappings

---

## 📥 How to Get Required Files

### Train Models Yourself

Run notebooks in order:

1. `preprocessing.ipynb`  
2. `eda.ipynb`  
3. `cnn_model.ipynb`  
4. `resnet_model.ipynb`  

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

pip install -r requirements.txt
streamlit run app.py
,,,
---
## 🌐 App Features

- 📤 Upload chest X-ray image  
- 🧾 Get prediction (COVID / Pneumonia / Normal)  
- 📊 View class probabilities  
- 🔥 Grad-CAM visualization  
- 🧠 Medical interpretation insights

## 🧠 Model Details

### CNN (Best Model)
- Custom architecture
- Trained from scratch
- Better Grad-CAM interpretability

### ResNet
- Transfer learning
- Used for comparison
## 🔥 Grad-CAM Explainability

Grad-CAM highlights regions influencing model decisions:

- 🔴 Red → High importance  
- 🔵 Blue → Low importance  

### Interpretation Patterns:

| Pattern | Meaning |
|--------|--------|
| Diffuse activation | COVID |
| Localized hotspot | Pneumonia |
| Uniform low activation | Normal |
| Boundary-focused | Structural (Normal) |

## 📊 Results

- Accuracy: ~94–98%
- High COVID recall
- CNN outperformed ResNet in interpretability

## ⚠️ Limitations

- Small dataset size  
- Possible model bias  
- Grad-CAM may highlight non-clinical features  
- Not suitable for real medical diagnosis

## 🧠 Key Insights

- Model differentiates based on spatial patterns  
- Diffuse vs localized activation is critical  
- Accuracy does not guarantee correct reasoning

## 🌐 Deployment

- Built using Streamlit  
- Deployable on Streamlit Cloud  

## 📌 Future Improvements

- Use larger datasets  
- Apply lung segmentation  
- Try advanced architectures (DenseNet, EfficientNet)  
- Perform clinical validation

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.  
It should **NOT be used for medical diagnosis**.

## 👨‍💻 Author

Jyoti Bhardwaj

## ⭐ Acknowledgements

- Kaggle Dataset Contributors  
- PyTorch  
- Streamlit  
- OpenCV  
