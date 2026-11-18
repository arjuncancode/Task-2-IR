# Comparative Study of Text Classification Algorithms

Name: __________ Register No.: __________ Course: __________ Date: __________

**Complete only after running the Python program. Replace every bracketed field with your actual output.**

## 1. Aim
Compare Decision Tree, KNN, SVM and Naïve Bayes on the same binary text classification problem.

## 2. Dataset and classes
20 Newsgroups: `rec.sport.baseball` = Baseball (class 0); `sci.space` = Space (class 1). Report total/train/test document counts from `outputs/run_metadata.json`.

## 3. Preprocessing and TF-IDF
Describe lowercase conversion, English stop-word removal, unigrams and bigrams, max_features=10000, min_df=2 and sublinear TF. State vocabulary size from metadata. Vectorizer is fitted on training data only.

## 4. Evaluation protocol
Stratified 75:25 train/test split, random_state=42, same split and same TF-IDF features for every model. Report accuracy and macro precision, recall and F1.

## 5. Actual results
Copy actual values from `outputs/model_metrics.csv`.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Decision Tree | [ ] | [ ] | [ ] | [ ] |
| KNN (k=5) | [ ] | [ ] | [ ] | [ ] |
| Linear SVM | [ ] | [ ] | [ ] | [ ] |
| Multinomial NB | [ ] | [ ] | [ ] | [ ] |

Insert `outputs/metric_comparison.png` and discuss observed differences.

## 6. Confusion matrix analysis
Use each `outputs/confusion_*.png` and `outputs/report_*.txt`. Explain correct predictions and false positives/negatives for both classes. Identify which class each model confuses more often. Do not guess without reading the matrices.

## 7. What-if experiment
Use `outputs/what_if_knn_k.csv` and `.png`. Compare K=3, 5 and 7. State how accuracy and macro F1 change, with actual values. Only K changes; split and TF-IDF stay fixed.

## 8. Discussion
Explain measured differences. Consider tree overfitting, KNN distance in sparse high-dimensional vectors, SVM margin separation, and Naïve Bayes' conditional-independence assumption. Link each explanation to a metric or confusion-matrix observation.

## 9. Best model and recommendation
Best by macro F1: [model and score]. Justify with actual metrics and class-wise errors. Mention trade-offs.

## 10. Conclusion and limitations
Summarize actual findings, KNN what-if result, and limitations of a two-category dataset and one holdout split.

## 11. Reproducibility
Python/package versions, date run, GitHub URL, and list the generated output files.
