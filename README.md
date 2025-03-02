# Predicting Student Dropout & Academic Success

## Overview
This project applies machine learning techniques to predict student dropout and academic success. By analyzing various academic, demographic, and socio-economic factors, the model helps identify at-risk students, enabling institutions to take proactive measures.

## Dataset
The dataset is sourced from the **UCI ML Repository** and consists of **37 features**, including:
- **Demographics**: Marital Status, Gender, Nationality
- **Academic Performance**: Admission grade, Curricular unit performance
- **Socio-economic Indicators**: Unemployment rate, Inflation rate, GDP
- **Target Variable**: `Target` (Dropout or Graduate)

## Project Structure
- **data.py**: Entry point for fetching the dataset, preprocessing, and initiating analysis & modeling.
- **dataanalysis.py**: Performs data cleaning, transformation, and normalization.
- **datamodelling.py**: Implements machine learning models for prediction.

## Installation

### Prerequisites
Ensure you have:
- Python 3.X

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Lekha03/02.-Predict-Students-Dropout-and-Academic-Success.git
   cd Predict_Student_Dropout
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python data.py
   ```

## Model Implementation
### 1. Data Processing
- Categorical variables were one-hot encoded.
- Numeric variables were standardized using `StandardScaler`.
- Feature selection was applied to remove low-impact attributes.

### 2. Machine Learning Models
- **Random Forest Classifier** (Primary Model)
- **Logistic Regression** (Baseline Model)
- **Support Vector Machine (SVM)** (Additional Evaluation)

### 3. Model Evaluation
- **Random Forest Classifier**
  - Confusion Matrix: [[517, 148], [83, 580]]
  - Accuracy: **72.3%**
  - Key Features: Admission Grade, Curricular Performance, Age at Enrollment

## Key Findings
- **Academic Performance** plays the most significant role in student success.
- **Age at Enrollment** impacts dropout rates.
- **Socio-economic factors** (Unemployment Rate, GDP) show moderate influence.
- **Random Forest outperformed other models** in predictive accuracy.

## Future Enhancements
- Integration of additional socio-economic datasets.
- Experimentation with deep learning models.
- Development of a real-time prediction system.

## License
This project is licensed under the **MIT License**.

## Contact
For questions or contributions, feel free to reach out or open an issue on the repository.

