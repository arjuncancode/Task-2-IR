from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, ConfusionMatrixDisplay

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"; OUT.mkdir(exist_ok=True)
CATS=["rec.sport.baseball","sci.space"]; NAMES=["Baseball","Space"]; SEED=42

def scores(y,p):
    pr,re,f1,_=precision_recall_fscore_support(y,p,average="macro",zero_division=0)
    return {"Accuracy":accuracy_score(y,p),"Precision_macro":pr,"Recall_macro":re,"F1_macro":f1}

def main():
    ds=fetch_20newsgroups(subset="all",categories=CATS,remove=("headers","footers","quotes"),shuffle=True,random_state=SEED)
    xtr,xts,ytr,yts=train_test_split(ds.data,ds.target,test_size=.25,random_state=SEED,stratify=ds.target)
    vec=TfidfVectorizer(lowercase=True,stop_words="english",ngram_range=(1,2),max_features=10000,min_df=2,sublinear_tf=True)
    Xtr=vec.fit_transform(xtr); Xts=vec.transform(xts)
    models={
      "Decision Tree":DecisionTreeClassifier(random_state=SEED,class_weight="balanced"),
      "KNN k=5":KNeighborsClassifier(n_neighbors=5,metric="cosine",algorithm="brute"),
      "Linear SVM":LinearSVC(C=1.0,random_state=SEED),
      "Multinomial NB":MultinomialNB(alpha=1.0)}
    rows=[]
    for name,m in models.items():
        m.fit(Xtr,ytr); pred=m.predict(Xts); row={"Model":name,**scores(yts,pred)}; rows.append(row)
        cm=confusion_matrix(yts,pred,labels=[0,1])
        ConfusionMatrixDisplay(cm,display_labels=NAMES).plot(cmap="Blues",values_format="d")
        plt.title("Confusion matrix - "+name); plt.tight_layout()
        plt.savefig(OUT/("confusion_"+name.lower().replace(" ","_")+".png"),dpi=180); plt.close()
        (OUT/("report_"+name.lower().replace(" ","_")+".txt")).write_text(classification_report(yts,pred,target_names=NAMES,zero_division=0))
    result=pd.DataFrame(rows).sort_values("F1_macro",ascending=False)
    result.to_csv(OUT/"model_metrics.csv",index=False)
    ax=result.set_index("Model")[["Accuracy","Precision_macro","Recall_macro","F1_macro"]].plot(kind="bar",ylim=(0,1),figsize=(10,6),rot=15)
    ax.set_ylabel("Score"); ax.set_title("Classifier comparison"); ax.grid(axis="y",alpha=.3); plt.tight_layout()
    plt.savefig(OUT/"metric_comparison.png",dpi=180); plt.close()
    wf=[]
    for k in [3,5,7]:
        m=KNeighborsClassifier(n_neighbors=k,metric="cosine",algorithm="brute"); m.fit(Xtr,ytr); pred=m.predict(Xts)
        wf.append({"K":k,**scores(yts,pred)})
    w=pd.DataFrame(wf); w.to_csv(OUT/"what_if_knn_k.csv",index=False)
    ax=w.set_index("K")[["Accuracy","F1_macro"]].plot(marker="o",ylim=(0,1),figsize=(8,5))
    ax.set_xlabel("K neighbours"); ax.set_ylabel("Score"); ax.set_title("KNN what-if experiment"); ax.grid(alpha=.3); plt.tight_layout()
    plt.savefig(OUT/"what_if_knn_k.png",dpi=180); plt.close()
    meta={"dataset":"20 Newsgroups","categories":CATS,"class_names":NAMES,"total_documents":len(ds.data),"train_documents":len(xtr),"test_documents":len(xts),"split":"75/25 stratified, random_state=42","tfidf":"lowercase, English stop words, 1-2 grams, max_features=10000, min_df=2, sublinear_tf; fit on train only","vocabulary_size":Xtr.shape[1],"metrics":"macro precision/recall/F1 plus accuracy"}
    (OUT/"run_metadata.json").write_text(json.dumps(meta,indent=2))
    (OUT/"run_summary.txt").write_text("ACTUAL RESULTS FROM THIS RUN\n"+result.to_string(index=False)+"\n\nKNN WHAT-IF\n"+w.to_string(index=False))
    print(result.to_string(index=False)); print("\nKNN what-if:\n",w.to_string(index=False)); print("\nOutputs saved in",OUT)

if __name__=="__main__": main()
