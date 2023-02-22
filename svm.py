import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay


data = pd.read_csv("creditcard.csv")
print(data.head())
X = data.drop(['Time','Class'],axis=1)
y = data['Class']
X = X[12300:12400]
y = y[12300:12400]
X['Amount'] = X['Amount'].apply(lambda x: (x - X['Amount'].mean())/X['Amount'].std())

print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
np.savetxt('confusion_matrix_svm.txt',cm, fmt='%d')
labels = [0,1]
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels =labels)
cm_display.plot()
plt.title('confusion matrix')
plt.savefig('confusion_matrix.png')
    
report =  classification_report(y_test,y_pred)

print(report)

with open('classification_report.txt','w') as file:
    file.write(report)
    
