from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier  
from XBNet.models import XBNETClassifier,XBNETRegressor

from XBNet.run import run_XBNET
from XBNet.training_utils import training,predict,test
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay,accuracy_score,f1_score,precision_score,recall_score
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc, plot_precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay,accuracy_score,f1_score,precision_score,recall_score
from sklearn.preprocessing import LabelEncoder

import pickle,os,csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Ensemble:
    def __init__(self):
        self.models = []
        # svm =  SVC(kernel='linear')
        # sgd = SGDClassifier(loss="log", penalty="l2")
        # knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
        # pac = PassiveAggressiveClassifier(C = 0.5, random_state = 5)
        # rf  = RandomForestClassifier(n_estimators= 10, criterion="entropy")  
        # dt  = DecisionTreeClassifier()
        svm = pickle.load(open('./result/svm/svm_model1.pkl', 'rb'))
        sgd = pickle.load(open('./result/sgd/sgd_model1.pkl', 'rb'))
        knn = pickle.load(open('./result/knn/knn1.pkl', 'rb'))
        pac = pickle.load(open('./result/passive_aggressive_clf/pac1.pkl', 'rb'))
        rf  = pickle.load(open('./result/random_forest/rf_model1.pkl', 'rb'))
        dt  = pickle.load(open('./result/decision_tree/dt_model1.pkl', 'rb'))
        xbnet_sigmoid = pickle.load(open('./result/xbnet/xbnet_model_layer=1_sigmoid.pkl', 'rb'))
        xbnet_softmax = pickle.load(open('./result/xbnet/xbnet_model_layer=1_softmax.pkl', 'rb'))
        xbnet_none = pickle.load(open('./result/xbnet/xbnet_model_layer=1_none.pkl', 'rb'))

        self.models.append(svm)
        self.models.append(sgd)
        self.models.append(knn)
        self.models.append(pac)
        self.models.append(rf)
        self.models.append(dt)
        self.models.append(xbnet_sigmoid)
        self.models.append(xbnet_softmax)
        self.models.append(xbnet_none)


    def predict(self,x_test):
        predicted = np.zeros((len(self.models),len(x_test)))
        n_estimators = len(self.models)-3
        for i in range(n_estimators):
            predicted[i] = self.models[i].predict(x_test) 

        predicted[n_estimators]= predict(self.models[-3],x_test.to_numpy())
        predicted[n_estimators+1]= predict(self.models[-2],x_test.to_numpy())
        predicted[n_estimators+2]= predict(self.models[-1],x_test.to_numpy())
        
        predicted = predicted.T
        predicted = predicted.tolist()
        
        consensus_values = [max(predicted[i],key=predicted[i].count) for i in range(len(predicted))]
        consensus_values = np.asarray(consensus_values)
        return consensus_values
        
    def calculate_metrices(self,y_test,y_pred):
        s=1
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
            
        row = ['Ensemble',"{:.4f}".format(acc),"{:.4f}".format(p),"{:.4f}".format(r),"{:.4f}".format(f1_scr),"{:.4f}".format(auc_precision_recall)]
        writer.writerow(row)
        file.close()

        my_path=format(os.getcwd())
        my_path=os.path.join(my_path,'result/ensemble')
        my_file = f'classification_report_ensemble{s}.txt'  #filename
        filename = os.path.join(my_path,my_file)

        with open(filename,'w') as file:
            file.write(report)
            file.write(f'\nensemble run number : {s}\n')
            file.write(f'\nTest accuracy : {acc}  |     f1-score(macro) : {f1_scr}\n')
            file.write(f'\nprecision     : {p}  |   recall  :  {r}\n ')
            file.write(f'\nArea under precision recall curve : {auc_precision_recall}')


        #confusion matrix
        cm = confusion_matrix(y_test,y_pred)
        labels = [0,1]
        cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels =labels)
        cm_display.plot()
        plt.title(f'confusion matrix ensemble{s}')
        my_file = f'confusion_matrix_ensemble{s}.png' #filename
        filename = os.path.join(my_path,my_file)
        plt.savefig(filename)
                

# # Create the individual models
# sgd = SGDClassifier(random_state=42)
# svm = SVC(kernel='linear', C=1.0, random_state=42)
# nb = GaussianNB()

# # Create the ensemble model
# ensemble = VotingClassifier(estimators=[('sgd', sgd), ('svm', svm), ('nb', nb)], voting='hard')

# # Fit the ensemble model to the training data
# ensemble.fit(X_train, y_train)

# # Make predictions on the test data
# y_pred = ensemble.predict(X_test)

# # Calculate the accuracy of the ensemble model
# accuracy = accuracy_score(y_test, y_pred)
# print("Ensemble accuracy:", accuracy)

if __name__=='__main__':
    data = pd.read_csv("creditcard.csv")
    X = data.drop(['Time','Class'],axis=1)
    Y = data['Class']
    #X = X.head(10000)
    #Y = Y.head(10000)
    X['Amount'] = X['Amount'].apply(lambda x: (x - X['Amount'].mean())/X['Amount'].std())

    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0)
    ensemble_model = Ensemble()
    y_pred = ensemble_model.predict(x_test)
    ensemble_model.calculate_metrices(y_test,y_pred)