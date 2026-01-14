# Implementation of Softmax Regression and Accelerated Gradient Descent

## Project Overview
This repository contains my implementation of a multi-class Softmax Regression model to classify spectral data. This project was completed as part of a Machine Learning university module.

The primary objective was to build the model and optimizer from first principles using Python and NumPy, rather than relying on high-level APIs like PyTorch or Scikit-Learn. This allowed for a deeper exploration of the underlying calculus and optimization mechanics.

## Mathematical Implementation
The project focuses on the manual translation of mathematical formulas into vectorized code:

* **Gradient Derivation:** Manually derived the gradient of the Cross-Entropy Loss function with respect to weights and implemented it using matrix operations.
* **Custom Optimizer:** Built an Accelerated Gradient Descent (AGD) algorithm from scratch. This includes an adaptive scaling mechanism (similar to RMSProp) where the update step is normalized by the second moment estimate.
* **Vectorization:** All linear algebra operations are fully vectorized in NumPy to ensure computational efficiency and avoid explicit Python loops.

## Methodology and Results
The project compares two distinct strategies for handling the training data to improve convergence:

1.  **Undersampling:** The initial approach used random undersampling. This resulted in slow convergence (taking approximately 100,000 iterations) and difficulty in separating the three distinct classes.
2.  **K-Means Initialization:** I implemented K-Means clustering to reduce dimensionality while preserving the structure of the spectral data.
    * **Outcome:** This method reduced the convergence time significantly (to approximately 40 iterations).
    * **Accuracy:** It successfully separated all three classes, correcting the class omission issues found in the undersampling approach.

## Dependencies
* Python 3.x
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn (used only for K-Means clustering comparison)
* SciPy
