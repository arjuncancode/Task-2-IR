# Comparative Text Classification Study

**Task:** classify text as Baseball (`rec.sport.baseball`) or Space (`sci.space`) using the public labelled 20 Newsgroups corpus.

## Run the experiment
Install Python 3.10+, then run:
```bash
python -m pip install -r requirements.txt
python src/compare_classifiers.py
```
The dataset is fetched by scikit-learn on first run, so internet is required initially.

## Fair protocol
One stratified 75:25 train/test split (`random_state=42`) is shared by all models. One TF-IDF matrix is used for all models: lowercase, English stop-word removal, unigrams+bigrams, max 10,000 features, min_df=2, sublinear TF. The vectorizer is fitted only on training text to avoid leakage.

Models: Decision Tree, cosine KNN (k=5), Linear SVM, Multinomial Naïve Bayes. Metrics: accuracy and macro precision, recall, F1. Confusion matrices and per-model classification reports are saved.

## Outputs created by your run
- `outputs/model_metrics.csv`
- `outputs/metric_comparison.png`
- `outputs/confusion_*.png`
- `outputs/report_*.txt`
- `outputs/what_if_knn_k.csv` and `.png` (K=3,5,7)
- `outputs/run_metadata.json`, `outputs/run_summary.txt`

**Important:** no scores are pre-filled. Run the program yourself and use its actual output in your report. The what-if experiment changes only K; dataset, split and features remain fixed.

## Model reasoning to test against results
Decision Trees can overfit sparse feature spaces; KNN depends on similarity and neighbour count; linear SVMs can separate high-dimensional sparse text effectively; Multinomial NB is a fast probabilistic text baseline with a conditional-independence assumption. These are explanations to assess against your measured results, not claims about which model will win.

This scaffold was created with AI assistance. Follow your institution's AI disclosure rules, understand the code and write your analysis based on your own run.
