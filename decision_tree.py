import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle,os,csv
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report,r2_score,mean_absolute_error,mean_squared_error,mean_squared_log_error
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay,accuracy_score,f1_score
from sklearn.tree import DecisionTreeClassifier , export_graphviz


s = 1


data = pd.read_csv("creditcard.csv")

X = data.drop(['Time','Class'],axis=1)
Y = data['Class']
#X = X.head(100)
#Y = Y.head(100)
X['Amount'] = X['Amount'].apply(lambda x: (x - X['Amount'].mean())/X['Amount'].std())

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)
clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)

#save model
filename = f'./result/decision_tree/dt_model{s}.pkl'
file = open(filename,'wb')
pickle.dump(clf,file)
file.close()
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
y_pred = loaded_model.predict(x_test)

#classification report
report = classification_report(y_test,y_pred)
print(report)
my_path=format(os.getcwd())
my_path=os.path.join(my_path,'result/decision_tree')
my_file = f'classification_report_dt{s}.txt'  #filename
filename = os.path.join(my_path,my_file)
acc = accuracy_score(y_test , y_pred)
f1_scr = f1_score(y_test , y_pred , average='macro')
print(f'Test accuracy : {acc}  | f1-score(macro) : {f1_scr}')

with open(filename,'w') as file:
    file.write(report)
    file.write(f'\ndecision tree run number : {s}')
    file.write(f'\nTest accuracy : {acc}  | f1-score(macro) : {f1_scr}')



#confusion matrix
cm = confusion_matrix(y_test,y_pred)
labels = [0,1]
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels =labels)
cm_display.plot()
plt.title(f'confusion matrix dt{s}')
my_file = f'confusion_matrix_dt{s}.png' #filename
filename = os.path.join(my_path,my_file)
plt.savefig(filename)
"""
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(f'./result/decision_tree/decision_tree.png')
Image(graph.create_png())"""