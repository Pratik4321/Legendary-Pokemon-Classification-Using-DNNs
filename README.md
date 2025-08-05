
# Legendary Pokémon Classification using Deep Neural Networks

This project classifies Pokémon as Legendary or non-Legendary using a ResNet-inspired deep neural network built in PyTorch. The model is trained on structured tabular data containing base stats and type features. The goal is to compare the performance of this deep learning approach with classical machine learning models on a binary classification task.

## 🔍 Overview

- Task: Binary classification (Legendary = 1, Non-Legendary = 0)
- Dataset: Pokémon with Stats (from Kaggle)
- Input Features: HP, Attack, Defense, Speed, Generation, Type 1, Type 2
- Target: Legendary
- Frameworks: PyTorch, scikit-learn, pandas, matplotlib

## 🧠 Models Implemented

- Logistic Regression (baseline)
- Random Forest Classifier
- Shallow Neural Network (1 hidden layer)
- Deep Feedforward Neural Network (3 layers)
- ResNet-style DNN (with residual blocks and dropout)

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score

✅ Best Performance:  
ResNet DNN — 93.1% Test Accuracy  
Outperformed all other models in F1-score and generalization


## 🛠 How to Run

1. Clone the repository:
   git clone https://github.com/Pratik4321/Legendary-Pokemon-Classification-Using-DNN

2. Install dependencies:
   pip install pandas numpy torch matplotlib scikit-learn

3. Place Pokemon.csv inside the working directory.

4. Run the script or Jupyter notebook to train and evaluate the models.

## 📎 References

- Dataset: https://www.kaggle.com/datasets/abcsds/pokemon
© 2025 Pradip Giri — Algoma University
  © 2025 Pratik Giri — Algoma University 
- PyTorch: https://pytorch.org  
- Scikit-learn: https://scikit-learn.org
