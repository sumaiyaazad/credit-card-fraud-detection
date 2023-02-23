import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from XBNet.training_utils import training,predict,test
from XBNet.models import XBNETClassifier,XBNETRegressor
from XBNet.run import run_XBNET
import pickle,os

data = pd.read_csv("creditcard.csv")

x_data = data.drop(['Time','Class'],axis=1)
y_data = data['Class']
#x_data = x_data.head(100)
#y_data = y_data.head(100)
x_data['Amount'] = x_data['Amount'].apply(lambda x: (x - x_data['Amount'].mean())/x_data['Amount'].std())
le = LabelEncoder()
y_data = np.array(le.fit_transform(y_data))
print(x_data.head())
X_train,X_test,y_train,y_test = train_test_split(x_data.to_numpy(),y_data,test_size = 0.2,random_state = 0)
X_train,X_valid,y_train,y_valid = train_test_split(x_data.to_numpy(),y_data,test_size = 0.1,random_state = 0)
layer = 1



model = XBNETClassifier(X_train,y_train,layer)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

m,acc, lo, val_ac, val_lo = run_XBNET(X_train,X_valid,y_train,y_valid,model,criterion,optimizer,32,20)

params = m.get_params()

my_path = format(os.getcwd())
my_path = os.path.join(my_path,'result/xbnet')
my_file = f'xbnet_model_layer={params[0]}_{params[1]}.pkl'
filename = os.path.join(my_path,my_file)
file = open(filename,'wb')
pickle.dump(m,file)
file.close()

loaded_model = pickle.load(open(filename,'rb'))
print('---test form our code --')
test(loaded_model,X_test,y_test)
"""print('------------------')
print(x_data.to_numpy()[0,:])
print('------------')
print(predict(m,x_data.to_numpy()))"""
