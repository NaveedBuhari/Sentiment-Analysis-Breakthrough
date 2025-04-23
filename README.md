# 💬 Sentiment Analysis Breakthrough

A machine learning mini-project that performs sentiment analysis on e-commerce product reviews. This project leverages NLP techniques to classify customer reviews as positive or negative, providing valuable insights into user satisfaction and product quality.

## 🧠 Objective

To implement a sentiment classification model using real-world product review data, and evaluate its performance using standard classification metrics and ROC curves.

## 📁 Project Structure

Sentiment Analysis Breakthrough/ │ ├── ML_Mini_Project.ipynb # Main Jupyter notebook with code and results ├── ML Report.pdf # Final report detailing methodology and findings ├── flipkart_data.xlsx # Dataset of customer reviews └── ROC curves/ # Stored ROC curve images for various models


## 🔍 Dataset

- Source: Flipkart product reviews (manually collected/simulated)
- Format: `.xlsx` spreadsheet
- Fields: Review text, Sentiment label (positive/negative)

## ⚙️ Technologies Used

- Python (Jupyter Notebook)
- Scikit-learn
- Pandas, NumPy
- NLTK / TextBlob (optional for preprocessing)
- Matplotlib / Seaborn for ROC visualizations

## 📊 Models Trained

- Logistic Regression
- Support Vector Machines (SVM)
- Naive Bayes
- Random Forest

> All models were evaluated using Accuracy, Precision, Recall, F1 Score, and AUC-ROC metrics.

## 📈 Visualizations

- ROC curves for model comparison stored in the `ROC curves/` directory
- Confusion matrix and performance metrics displayed in the notebook

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/NaveedBuhari/Sentiment-Analysis-Breakthrough.git
   cd Sentiment-Analysis-Breakthrough/Sentiment Analysis Breakthrough

2. Open the notebook:
- Using Jupyter Notebook or Google Colab
- File: ML_Mini_Project.ipynb

3. Run all cells to train models and view results

> **🔔 Note:**  
> This project was developed as part of a machine learning mini-project and is intended for educational and research purposes. The dataset used is manually collected/simulated and may not reflect actual Flipkart reviews. For production-grade applications, further improvements in data quality, preprocessing, and model tuning are recommended.

   
