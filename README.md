# Sickle Cell Crisis Forecaster

**Leveraging AI to Predict Acute and Chronic Pain Crises in Sickle Cell Disease Patients Using Symptoms and Vital Signs**  
Author: *Julia Akuorkor Mc-Addy*  
Institution: *Ashesi University*  
Year: *2025*

---

## Project Overview

This project investigates the use of Artificial Intelligence (AI) to predict acute and chronic pain crises in Sickle Cell Disease (SCD) patients. It uses a clinically-informed **synthetic dataset (~55,000 samples)**  to simulate real-world complexity.  

Two models were developed:
- An **ensemble machine learning model** (Random Forest, Gradient Boosting, XGBoost)
- A **mechanistic ODE model** using PyTorch and a differentiable Runge-Kutta solver

Both models achieved high predictive accuracy (~0.97) and strong ROC AUC (~0.99), supporting AI's potential to assist in early medical intervention for SCD.

---

## Deployed Application

An interactive aaplication is available on **Hugging Face Spaces**, built with Streamlit and PyTorch.

 **[Launch the App](https://huggingface.co/spaces/JuliaMc/scd_crisis_forecaster)**  
Users can input patient vitals and symptoms to receive time-resolved crisis probability predictions at 1h, 6h, 12h, and 24h intervals.

---

## GitHub Repository

Source code, models, synthetic data, and other resources:  
 **[GitHub Repository](https://github.com/Julia-Mc-Addy/SCD_Pain_Crises_Prediction/tree/main)**

---
