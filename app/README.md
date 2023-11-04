# Breast Cancer Prediction using Naive Bayes

## Overview

This project is developed by Elijah C Walker for the ANA680 Machine Learning Model Deployment course at National University. It utilizes a Naive Bayes classifier to predict the malignancy of breast tumors with an accuracy of 93.7%. The classifier makes predictions based on four key features that exhibit the highest correlation with the tumor class.

## Features Used for Prediction

The model uses the following features from the Breast Cancer Wisconsin (Original) Dataset:

-   Clump Thickness (`clump_thickness`)
-   Uniformity of Cell Size (`uniform_cell_size`)
-   Uniformity of Cell Shape (`uniform_cell_shape`)
-   Bland Chromatin (`bland_chromatin`)

## Tools and Libraries

The model was developed using the following tools and libraries:

-   Google Colab: For writing and executing Python in an interactive environment.
-   Pickle: To serialize the model object for later use.
-   NumPy: For efficient numerical computations.
-   scikit-learn (`sklearn`): For implementing the Naive Bayes algorithm.
-   pandas: For data manipulation and analysis.

## Model Training

## Model Training

The Naive Bayes model was trained on the Breast Cancer Wisconsin (Original) Dataset using `scikit-learn`. The dataset was processed and split into training and test sets. The features selected for the model were determined by their correlation with the class label, ensuring the most relevant predictors were used.
