# 🧠 Synthetic Brain Tumor Image Generation Using Generative Adversarial Network (DC-GAN + CNN)

## 🧾 Introduction
This project explores the use of a Deep Convolutional Generative Adversarial Network (DC-GAN) for synthetic brain tumor MRI image generation and evaluates its effectiveness in augmenting a CNN-based brain tumor classification system. The GAN is trained on a curated dataset of real brain MRI scans to learn and replicate tumor patterns. The generated synthetic images are then used alongside real images to enhance classification accuracy. This approach addresses data scarcity in medical imaging and contributes to building more robust deep learning models for diagnostics.

---

## 📂 Project Metadata

### 👨‍💻 Authors
- **Team:** [Your Name]
- **Supervisor:** [Supervisor Name]
- **Affiliation:** [Your Institution / Department]

### 📁 Project Files
- **Presentation:** `presentation.pptx`
- **Report:** `report.pdf`

### 📄 Reference Papers
- [Kazeminia et al., GAN-Based Synthetic Brain MR Image Generation](https://arxiv.org/abs/2001.06993)
- [Salehinejad et al., GANs for Medical Imaging](https://arxiv.org/abs/1806.01313)

### 🧠 Dataset
- [Kaggle: Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

---

## ⚙️ Project Technicalities

### 📚 Key Terminologies
- **GAN / DC-GAN** – A model that learns to generate realistic images using a generator-discriminator architecture.
- **CNN** – A deep neural network used for image classification.
- **Tumor Recall** – A measure of how well tumor cases are identified.
- **Data Augmentation** – Techniques used to increase training data size and diversity.

### 🧩 Problem Statements
- Limited dataset for training deep models.
- Traditional augmentation lacks structural tumor variability.
- Image quality variance limits classification reliability.

### 💡 Research Ideas
1. **GAN-Based Augmentation** to generate more tumor images.
2. **Combined DC-GAN + CNN Pipeline** to improve classification.
3. **Visual + Quantitative Evaluation** to verify improvements.

---

## ✅ Proposed Solution: Summary
- **DC-GAN** generates synthetic grayscale tumor images from random noise.
- **CNN Classifier** is trained using a mix of real and generated images.
- **Evaluation** is done using accuracy, recall, confusion matrix, and qualitative image inspection.

---

## 🧱 Project Structure

- `dcgan_train.ipynb` – GAN training with image generation.
- `cnn_classifier.ipynb` – Tumor vs no-tumor classification.
- `metrics.py` – Metrics and plots (accuracy, loss, confusion matrix).
- `visualization.py` – Plots generated images across epochs.

---

## 🔁 Model Workflow

### Input
- Brain MRI dataset with 253 grayscale images (155 tumor, 98 no tumor).

### GAN Training
- Generator learns to synthesize 64x64 tumor-like MRIs.
- Discriminator distinguishes real vs fake images.
- Training done over 10 epochs.

### CNN Classification
- Dataset includes real + generated images.
- Split: 80% training, 20% testing.
- Trained for 50 epochs with data augmentation (flipping, shifting, dropout).
- Metrics collected: accuracy, recall, confusion matrix, F1-score.

---

## 🚀 How to Run

### 1. Clone the Repo
```bash
git clone https://github.com/yourusername/synthetic-brain-mri-gan.git
cd synthetic-brain-mri-gan
