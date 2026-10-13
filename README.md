# Email Spam Classification Using Natural Language Processing

## Project Overview

This project develops a Natural Language Processing (NLP) system for classifying email messages as either **legitimate** or **spam**.

The system is developed using the **Enron Spam Dataset** and applies text preprocessing, TF-IDF feature extraction, and a Multinomial Naive Bayes classifier.

The project also evaluates the model using progressively larger subsets of the dataset to investigate how dataset size affects classification performance.

---

## Classification Labels

The dataset contains two target classes:

- `0` — Legitimate email
- `1` — Spam email

---

## Dataset

The project uses the **Enron Spam Dataset**, obtained from Kaggle.

Dataset used:

**Enron Spam — Bert van Keulen**

The original dataset contained approximately 33,716 labelled emails.

Duplicate email messages were removed before model development to reduce the possibility of identical emails appearing in both the training and testing datasets.

After duplicate removal, the working dataset contained:

**30,494 unique emails**

The dataset itself is not included in this repository.

To reproduce the project, download `Enron_Txt_fn.csv` from the Kaggle Enron Spam dataset and make it available to the notebook.

---

## Project Workflow

The NLP pipeline follows the steps below:

```text
Raw Email Dataset
        ↓
Dataset Inspection
        ↓
Duplicate Handling
        ↓
Basic Text Cleaning
        ↓
Stop-word Removal
        ↓
Lemmatization
        ↓
Train/Test Split
        ↓
TF-IDF Text Encoding
        ↓
Multinomial Naive Bayes
        ↓
Model Evaluation
        ↓
Spam / Legitimate Prediction
        ↓
Progressive Dataset Size Experiments
```

---

## Text Preprocessing

The email text is prepared using the following preprocessing operations:

1. Conversion to lowercase
2. Removal of unnecessary formatting characters
3. Removal of punctuation and special characters
4. Removal of unnecessary whitespace
5. Tokenization
6. English stop-word removal
7. Lemmatization

The preprocessing is implemented using Python, regular expressions, and NLTK.

---

## Train/Test Split

The processed dataset is divided into:

- **80% training data**
- **20% testing data**

A stratified split is used to preserve approximately the same proportion of spam and legitimate emails in both datasets.

The test dataset remains separate from the training process and is used to evaluate the model on previously unseen emails.

---

## TF-IDF Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert the processed email text into numerical features that can be used by the machine-learning classifier.

The TF-IDF vectorizer is fitted only on the training data to avoid data leakage.

The same fitted vectorizer is then used to transform the test data and new email messages.

---

## Machine-Learning Model

The classification model used in this project is:

**Multinomial Naive Bayes**

The model is trained using:

- TF-IDF features from the training emails
- Corresponding spam/legitimate labels

The trained model predicts whether previously unseen emails belong to the legitimate or spam class.

---

## Model Evaluation

The classifier is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Final Test Result

Using the full working dataset, the model achieved:

**Accuracy: 98.79%**

The confusion matrix was:

```text
[[3162   20]
 [  54 2863]]
```

This represents:

- 3,162 legitimate emails correctly classified as legitimate
- 20 legitimate emails incorrectly classified as spam
- 54 spam emails incorrectly classified as legitimate
- 2,863 spam emails correctly classified as spam

The model correctly classified:

**6,025 out of 6,099 test emails**

---

## Progressive Dataset Experiments

The classifier was evaluated using progressively larger datasets.

| Dataset Size | Accuracy | Precision | Recall | F1 Score |
|-------------:|---------:|----------:|-------:|---------:|
| 1,000 | 95.00% | 95.28% | 95.00% | 94.98% |
| 5,000 | 97.70% | 97.70% | 97.70% | 97.70% |
| 10,000 | 98.40% | 98.40% | 98.40% | 98.40% |
| 20,000 | 98.15% | 98.16% | 98.15% | 98.15% |
| 30,494 | 98.79% | 98.79% | 98.79% | 98.79% |

Overall, the results show that classification performance generally improved as more data became available.

A small reduction in accuracy occurred between the 10,000-email and 20,000-email experiments, demonstrating that model performance does not necessarily increase monotonically with dataset size.

The highest accuracy was obtained using the complete working dataset.

---

## Predicting New Emails

The project includes a prediction function that accepts raw email text and passes it through the same preprocessing and TF-IDF pipeline used during model development.

Example:

```python
email = """
Congratulations!

You have won a cash prize.
Click the link immediately to claim your reward.
"""

print(predict_email(email))
```

Example output:

```text
Spam
```

A legitimate example:

```python
email = """
Hello John,

Please review the project document before tomorrow's meeting.

Regards,
David
"""

print(predict_email(email))
```

The system returns either:

```text
Spam
```

or:

```text
Legitimate
```

---

## Technologies Used

- Python
- Pandas
- NLTK
- Scikit-learn
- Matplotlib
- Joblib
- Google Colab

---

## Repository Structure

```text
enron-spam-classification/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── enron_spam_classification.ipynb
│
├── results/
│   ├── dataset_size_experiment_results.csv
│   ├── dataset_size_vs_accuracy.png
│   └── confusion_matrix.png
│
├── models/
│   ├── spam_classifier.pkl
│   └── tfidf_vectorizer.pkl
│
└── report/
    └── project_report.pdf
```

---

## Running the Project

### 1. Clone or download the repository

Download the repository to your computer or open the notebook using Google Colab.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

When using Google Colab, most required libraries are already available.

### 3. Download the dataset

Download the Enron Spam dataset from Kaggle.

Place:

```text
Enron_Txt_fn.csv
```

in the location expected by the notebook.

### 4. Run the notebook

Run the notebook cells sequentially.

The notebook performs:

- Dataset inspection
- Text cleaning
- NLP preprocessing
- TF-IDF encoding
- Model training
- Model evaluation
- New email prediction
- Progressive dataset experiments
- Result visualization

---

## Results

The final Multinomial Naive Bayes classifier achieved approximately:

**98.79% test accuracy**

on the full working dataset.

The progressively larger dataset experiments showed an overall improvement in classification performance as additional training data became available.

---

## Limitations

Although the classifier achieved strong performance on the Enron dataset, its performance should not automatically be assumed to be the same for emails from other domains.

The model relies primarily on statistical patterns in TF-IDF features rather than a semantic understanding of email content.

Very short or unusual messages may therefore occasionally be misclassified.

For example, short messages may contain very few useful features after stop-word removal.

---

## Conclusion

This project demonstrates how Natural Language Processing and supervised machine learning can be used to distinguish spam emails from legitimate emails.

The system combines text preprocessing, TF-IDF feature extraction, and Multinomial Naive Bayes classification.

The final model achieved 98.79% accuracy on the held-out test dataset, while experiments using progressively larger datasets demonstrated the effect of dataset size on classification performance.

The results show that a classical NLP pipeline can provide strong performance for binary email spam classification.