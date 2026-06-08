# E-Commerce Customer Segmentation using K-Means Clustering

## Project Overview
This project implements an end-to-end unsupervised machine learning pipeline to segment unique e-commerce customers based on their purchasing footprints. Utilizing the massive UCI Online Retail II dataset containing raw transaction records, I engineered distinct behavioral metrics using an RFM (Recency, Frequency, Monetary) Analysis framework and applied K-Means Clustering to isolate distinct customer personas for targeted marketing.

---

## Tech Stack & Skills
* Language: Python
* Data Engineering: Pandas, NumPy
* Machine Learning: Scikit-Learn (KMeans, StandardScaler, silhouette_score)
* Data Visualization: Matplotlib, Seaborn, Mplot3d (3D Projection)
* Concepts: Data Preprocessing, Log-Transformations (Skewness Correction), Feature Scaling, Hyperparameter Tuning

---

## Dataset Source
The analysis uses the actual transactional logs of a UK-based non-store online retail business. Due to file size constraints (>90 MB), the raw dataset is not hosted directly inside this repository.

* Download Link: [Kaggle - Online Retail II Dataset](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci/data)
* Setup: Download online_retail_II.csv and place it directly into your root directory before running the notebook.

---

## Data Pipeline Workflow

### 1. Data Cleaning & Guardrails
* Handled missing categorical data in the Customer ID column without altering global patterns.
* Filtered transactional cancellations (invoices prefixed with 'C') and eliminated anomalies with non-positive quantities or prices.
* Generated total expenditure calculations using the explicit Quantity and Price features.

### 2. Feature Engineering (RFM Framework)
Collapsed individual transaction lines into unique customer profiles calculated against a baseline temporal snapshot:
* Recency (R): Days elapsed since the customer's absolute last purchase.
* Frequency (F): Count of unique, successful invoices generated.
* Monetary (M): Aggregate gross expenditure.

### 3. Preprocessing for Spatial Distance Math
* Mathematical Skewness Correction: Applied an element-wise Log-transformation to resolve extreme right-skewed business revenue distributions.
* Standardization: Utilized StandardScaler to align features to a standard normal distribution (mean=0, variance=1), preventing scale disparities from biasing the Euclidean distance modeling.

---

## Hyperparameter Tuning & Model Performance

### The Elbow Method
By evaluating the Within-Cluster Sum of Squares (WCSS) across a range of clusters, the notebook plots the metric curve to identify the sharp inflection point (the "elbow"), establishing the mathematical boundary where adding further clusters yields diminishing returns.

### Silhouette Validation
The pipeline utilizes the Global Silhouette Coefficient to mathematically validate the density and separation quality of the generated clusters, ensuring clear boundaries amidst inherently noisy, real-world retail data.

---

## How to Run the Project

1. Clone this repository to your local system:
   ```bash
   git clone [https://github.com/yourusername/customer-segmentation-kmeans.git](https://github.com/yourusername/customer-segmentation-kmeans.git)
