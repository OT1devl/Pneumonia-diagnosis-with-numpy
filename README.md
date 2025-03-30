<h1 align="center">Pneumonia diagnosis</h1>
<h3 align="center">Implementation with NumPy</h3>

<p align="center">
  <img src="https://img.shields.io/github/languages/top/OT1devl/Pneumonia-diagnosis-with-numpy?style=flat" alt="Languages" />
  <img src="https://img.shields.io/github/last-commit/OT1devl/Pneumonia-diagnosis-with-numpy?style=flat" alt="Last Commit" />
</p>
## 🚀 **Project Overview**
This project implements a machine learning model to diagnose pneumonia from X-ray images, using NumPy as the primary data processing library.

## 📚 **Table of Contents**
- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contact](#contact)

---

## 🧑‍⚕️ **Introduction**
Pneumonia is an infection that inflames the air sacs in one or both lungs. The goal of this project is to facilitate automated pneumonia diagnosis by analyzing X-ray images using **NumPy-based** data processing techniques. This is a **from-scratch** model implemented purely with NumPy, intended for **educational purposes** only, and **not** for real-world use.

This project is meant to demonstrate how machine learning can be implemented **without relying on external libraries**, showcasing fundamental concepts using only NumPy. It is **not designed for contributions or extensions**.

---

## ⚡ **Features**
- **Image Processing:** Uses NumPy for handling and analyzing X-ray images.  
- **Custom Models:** Implements machine learning models without relying on high-level frameworks.  
- **Performance Evaluation:** Includes metrics to assess model accuracy and efficiency.

---

## 🛠️ **Prerequisites**
Before running the project, ensure you have the following Python packages installed:

- `NumPy`  
- `Matplotlib`  
- `cv2` (OpenCV)

---

## 📥 **Installation**

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/OT1devl/Pneumonia-diagnosis-with-numpy.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Pneumonia-diagnosis-with-numpy
   ```

3. (Optional) Create a virtual environment to isolate dependencies:

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

4. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Usage

1. Ensure that X-ray images are located in the appropriate directory within the `datasets` folder.  

2. Run the training notebook to train the model:

   ```bash
   jupyter notebook train.ipynb
   ```

3. Once trained, you can test the model's performance using the test notebook:

   ```bash
   jupyter notebook test.ipynb
   ```

## 🗂️ Project Structure

- `datasets/`: Contains X-ray images used for training and testing.  
- `functions.py`: Defines helper functions for data processing and model operations.  
- `models.py`: Implements the machine learning model.  
- `optimizers.py`: Includes optimization algorithms for training.  
- `utils.py`: Provides utility functions for various project tasks.  
- `train.ipynb`: Notebook for training the model.  
- `test.ipynb`: Notebook for evaluating the trained model.  

## 📧 Contact

For any questions or suggestions, please contact me at:  
📧 [otidevv1@gmail.com](mailto:otidevv1@gmail.com)  
🌐 Visit my [GitHub profile](https://github.com/OT1devl).

