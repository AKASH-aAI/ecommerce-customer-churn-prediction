# E-Commerce Customer Churn Prediction System

A Machine Learning-powered web application that predicts customer churn probability and identifies customers who may be at risk of leaving an e-commerce platform.

**Built with Python, Scikit-learn, Random Forest and Flask.**

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black)](https://flask.palletsprojects.com/)
[![Status](https://img.shields.io/badge/Status-Development-yellow)]()

## 🌐 Live Application

**Live Demo:** https://ecommerce-customer-churn-prediction-5ejh.onrender.com

**GitHub Repository:** This repository.

---

## 1. Problem Statement

Customer churn is a major challenge for e-commerce businesses. When customers stop purchasing or engaging with a platform, businesses may lose revenue and long-term customer relationships.

Identifying customers who are likely to churn can help businesses take proactive retention measures.

The objective of this project is to develop a Machine Learning system that:

- Analyses customer behaviour and purchasing patterns.
- Predicts the probability of customer churn.
- Identifies high-risk and low-risk customers.
- Allows businesses to assess individual customers.
- Demonstrates how a trained ML model can be integrated into a usable web application.

The system is designed as a customer churn prediction prototype using historical customer behaviour data.

---

## 2. Dataset Information

This project uses the **E-Commerce Customer Behavior Dataset**, sourced from Kaggle.

- **Dataset size:** 50,000 customer records.
- **Target variable:** `Churned`
- **Prediction type:** Binary Classification.
- **Target values:** 0 = Not Churned, 1 = Churned.
- **Features:** 24 input features.

### Dataset Features

The dataset contains customer demographic, engagement, purchasing and behavioural information.

| Category | Features |
|---|---|
| Customer Profile | Age, Gender, Country, City |
| Membership | Membership_Years, Signup_Quarter |
| Engagement | Login_Frequency, Session_Duration_Avg, Pages_Per_Session |
| Shopping Behaviour | Cart_Abandonment_Rate, Wishlist_Items |
| Purchase History | Total_Purchases, Average_Order_Value, Days_Since_Last_Purchase |
| Marketing | Discount_Usage_Rate, Email_Open_Rate |
| Customer Experience | Returns_Rate, Customer_Service_Calls, Product_Reviews_Written |
| Digital Behaviour | Social_Media_Engagement_Score, Mobile_App_Usage |
| Financial Behaviour | Payment_Method_Diversity, Lifetime_Value, Credit_Balance |

**Important Dataset Note:**

The original dataset did not contain a unique Customer ID. Therefore, synthetic identifiers were generated for individual records to support customer lookup functionality in the application.

These identifiers are application-generated record identifiers, not genuine customer IDs.

The dataset represents customer behaviour records and should not be interpreted as a verified live e-commerce transaction database.

**Dataset Source:** [E-Commerce Customer Behavior Dataset – Kaggle](https://www.kaggle.com/datasets/dhairyajeetsingh/ecommerce-customer-behavior-dataset)

---

## 3. My Contribution & AI-Assisted Development

I approached this project in two parts: Machine Learning development and web application development.

### My Machine Learning Work

I independently worked on the Machine Learning workflow, including:

- Dataset exploration and understanding.
- Data cleaning and preprocessing.
- Numerical and categorical feature analysis.
- Missing-value handling.
- Feature preparation and encoding.
- Train-test splitting.
- Random Forest model training.
- Class imbalance handling using class weights.
- Model evaluation and confusion matrix analysis.
- Precision and Recall analysis.
- Classification threshold experimentation.
- Selecting a decision threshold for the application.

### AI-Assisted Development

I used **DeepSeek and AI coding tools** to accelerate the web application development.

AI assistance was primarily used for:

- Generating and refining Flask backend implementation.
- Creating application routes and form handling.
- Integrating the saved ML model with Flask.
- Developing HTML templates and CSS styling.
- Building the dashboard interface and responsive layouts.
- Implementing JavaScript interactions and Chart.js visualizations.
- Debugging frontend and backend integration issues.

The Machine Learning experimentation and modelling decisions were my own work. AI tools assisted with application implementation and coding productivity.

I believe AI is a valuable development assistant when combined with technical understanding, independent decision-making and proper testing.

---

## 4. Machine Learning Workflow

### Algorithm: Random Forest Classifier

Random Forest was selected for this binary classification problem.

The model uses multiple decision trees and combines their predictions to classify customers based on their behavioural features.

### Preprocessing

The preprocessing workflow includes:

1. Separating numerical and categorical features.
2. Handling missing numerical values using median imputation.
3. Handling missing categorical values using most-frequent imputation.
4. Applying One-Hot Encoding to categorical variables.
5. Combining preprocessing and classification into a Scikit-learn Pipeline.

### Model Configuration

| Parameter | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Number of Estimators | 200 |
| Maximum Depth | 15 |
| Class Weight | Balanced |
| Random State | 42 |

### Model Evaluation

Previously reported notebook results:

| Metric | Result |
|---|---:|
| Training Accuracy | 96.51% |
| Testing Accuracy | 91.07% |
| ROC-AUC | 0.8794 |

Additional evaluation included:

- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report
- Threshold Analysis

### Classification Threshold

The default classification threshold was experimentally adjusted.

The current application uses a threshold of **0.43**.

A customer is classified as:

- **High Risk:** Predicted churn probability is greater than or equal to 43%.
- **Low Risk:** Predicted churn probability is below 43%.

The threshold is configurable through a saved model configuration file.

**Model Evaluation Limitation:** The earlier preprocessing workflow performed missing-value imputation before the train-test split. This may introduce data leakage. The reported metrics are preliminary and should not be treated as final generalisation performance until the model is retrained with preprocessing fitted exclusively on training data.

---

## 5. Application Features

### Existing Customer Search
- Search customers using their generated Customer ID.
- Retrieve existing customer information.
- Generate churn probability using the trained model.

### Manual Churn Prediction
- Enter customer information manually.
- Predict churn without requiring an existing Customer ID.
- Leave unknown fields blank.
- Handle missing values through the fitted ML pipeline.

### Prediction Results
- Display estimated churn probability.
- Classify customers into High Risk and Low Risk.
- Show the customer information used for prediction.
- Display the configured classification threshold.

### Dashboard
- Total customer count.
- Churned customer count.
- Active customer count.
- Overall historical churn rate.
- High-risk customer information.
- Customer-related KPIs.

### Model Insights
- Random Forest configuration.
- Model evaluation information.
- Feature importance visualization.

---

## 6. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing and Machine Learning |
| Random Forest | Churn classification |
| Joblib | Model serialization |
| Flask | Web application backend |
| HTML5 | Page structure |
| CSS3 | UI styling |
| JavaScript | Frontend interactions |
| Chart.js | Data visualization |

---

## 7. Project Structure

```text
E-commerce-Customer-Churn-Prediction/
│
├── app.py
├── requirements.txt
│
├── data/
│   ├── raw_data.csv
│   └── cleaned_customers.csv
│
├── models/
│   ├── churn_model.pkl
│   ├── threshold.pkl
│   └── model.ipynb
│
├── notebooks/
│   └── data_clean.ipynb
│
├── templates/
│   ├── base.html
│   └── index.html
│
└── static/
    ├── style.css
    └── script.js
```

---

## 8. Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ecommerce-customer-churn-prediction.git
```

### Step 2: Navigate to the Project

```bash
cd ecommerce-customer-churn-prediction
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Application

```bash
python app.py
```

Open your browser:

```text
http://127.0.0.1:5000
```

---

## 9. Future Improvements

- Leakage-free model retraining and validation.
- Hyperparameter optimization.
- Model performance improvement.
- Improved probability calibration.
- Additional customer retention insights.
- Cloud deployment and public accessibility.
- Expanded analytics and monitoring.

---

## 10. Project Disclaimer

This project is developed for educational and portfolio purposes.

The dataset contains customer behaviour records and generated identifiers. Model predictions are estimates based on the available dataset and should not be treated as guaranteed customer behaviour or used as the sole basis for real business decisions.

---

## Author

**Akash Chauhan**

Interested in Data Analytics, Data Science, Machine Learning and AI-assisted development.

- GitHub: [AKASH-aAI](https://github.com/AKASH-aAI)

**If you find this project useful, feel free to explore the repository and share your feedback!**
