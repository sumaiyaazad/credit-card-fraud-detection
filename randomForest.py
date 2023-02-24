import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay
import pickle,os,csv
from sklearn.metrics import classification_report,r2_score,mean_absolute_error,mean_squared_error,mean_squared_log_error
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc, plot_precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay,accuracy_score,f1_score,precision_score,recall_score

s = 1

data = pd.read_csv("creditcard.csv")
print(data.head())
X = data.drop(['Time','Class'],axis=1)
y = data['Class']
#X = X[12300:12400]
#y = y[12300:12400]
X['Amount'] = X['Amount'].apply(lambda x: (x - X['Amount'].mean())/X['Amount'].std())

print(X.head())
print(y.head())
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.30,random_state=0)
clf =  RandomForestClassifier(n_estimators= 10, criterion="entropy")  
clf.fit(X_train, y_train)


filename = f'./result/random_forest/rf_model{s}.pkl'
file = open(filename,'wb')
pickle.dump(clf,file)
file.close()
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
y_pred = loaded_model.predict(x_test)

#classification report
report = classification_report(y_test,y_pred)
print(report)

#calculate matrices
acc = accuracy_score(y_test , y_pred)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    
# Use AUC function to calculate the area under the curve of precision recall curve
auc_precision_recall = auc(recall, precision)

f1_scr = f1_score(y_test , y_pred , average='macro')
p = precision_score(y_test,y_pred,average ='macro')
r = recall_score(y_test,y_pred,average='macro')
print(f'Test accuracy : {acc}  | f1-score(macro) : {f1_scr} ')
print(f'Area under precision recall curve : {auc_precision_recall}')
print(f'p:{p} | r: {r}')

file = open('./result/model_result.csv','a')
writer=csv.writer(file,lineterminator='\n')
    
row = ['Random Forest',"{:.4f}".format(acc),"{:.4f}".format(p),"{:.4f}".format(r),"{:.4f}".format(f1_scr),"{:.4f}".format(auc_precision_recall)]
writer.writerow(row)
file.close()

my_path=format(os.getcwd())
my_path=os.path.join(my_path,'result/random_forest')
my_file = f'classification_report_rf{s}.txt'  #filename
filename = os.path.join(my_path,my_file)

with open(filename,'w') as file:
    file.write(report)
    file.write(f'\nrandom forest run number : {s}\n')
    file.write(f'\nTest accuracy : {acc}  |     f1-score(macro) : {f1_scr}\n')
    file.write(f'\nprecision     : {p}  |   recall  :  {r}\n ')
    file.write(f'\nArea under precision recall curve : {auc_precision_recall}')






#confusion matrix
cm = confusion_matrix(y_test,y_pred)
labels = [0,1]
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels =labels)
cm_display.plot()
plt.title(f'confusion matrix rf{s}')
my_file = f'confusion_matrix_rf{s}.png' #filename
filename = os.path.join(my_path,my_file)
plt.savefig(filename)
