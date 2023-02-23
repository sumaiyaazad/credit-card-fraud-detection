import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay

s = 2

data = pd.read_csv("creditcard.csv")
print(data.head())
X = data.drop(['Time','Class'],axis=1)
y = data['Class']
#X = X[12300:12400]
#y = y[12300:12400]
X['Amount'] = X['Amount'].apply(lambda x: (x - X['Amount'].mean())/X['Amount'].std())

print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)


filename = f'./result/svm/svm_model{s}.pkl'
file = open(filename,'wb')
pickle.dump(svclassifier,file)
file.close()
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
y_pred = loaded_model.predict(x_test)

#classification report
report = classification_report(y_test,y_pred)
print(report)
my_path=format(os.getcwd())
my_path=os.path.join(my_path,'result/svm')
my_file = f'classification_report_svm{s}.txt'  #filename
filename = os.path.join(my_path,my_file)
acc = accuracy_score(y_test , y_pred)
f1_scr = f1_score(y_test , y_pred , average='macro')
print('Test accuracy : ',acc,' | f1-score(macro) : ',f1_scr)

with open(filename,'w') as file:
    file.write(report)
    file.write(f'\nsvm run number : {s}')
    file.write(f'\nTest accuracy : {acc}  | f1-score(macro) : {f1_scr}')





#confusion matrix
cm = confusion_matrix(y_test,y_pred)
labels = [0,1]
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels =labels)
cm_display.plot()
plt.title(f'confusion matrix svm{s}')
my_file = f'confusion_matrix_svm{s}.png' #filename
filename = os.path.join(my_path,my_file)
plt.savefig(filename)
