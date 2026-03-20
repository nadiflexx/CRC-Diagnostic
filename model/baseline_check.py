import pandas as pd, os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score

p = os.path.join(os.path.dirname(__file__), '..', 'Data', 'processed', 'dataset_clinico_tumoral.csv')
df = pd.read_csv(p)
X = df.drop(columns=[c for c in ['Patient_ID','Diagnosis'] if c in df.columns])
y = df['Diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_prob = clf.predict_proba(X_test)[:,1]
y_pred = (y_prob>=0.5).astype(int)
print('Logistic Regression on test:')
print('Recall', recall_score(y_test, y_pred))
print('Precision', precision_score(y_test, y_pred))
print('F1', f1_score(y_test, y_pred))
print('ROC AUC', roc_auc_score(y_test, y_prob))
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
