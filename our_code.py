import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from XBNet.training_utils import training,predict,test
from XBNet.models import XBNETClassifier,XBNETRegressor
from XBNet.run import run_XBNET
import pickle

data = pd.read_csv('creditcard.csv')
print(data.shape)
x_data = data[data.columns[1:-1]].head(1000)
x_data['Amount'] = x_data['Amount'].apply(lambda x: (x - x_data['Amount'].mean())/x_data['Amount'].std())

print(x_data.head())

print(x_data.shape)
y_data = data[data.columns[-1]].head(1000)

print(y_data.shape)
le = LabelEncoder()
y_data = np.array(le.fit_transform(y_data))


X_train,X_test,y_train,y_test = train_test_split(x_data.to_numpy(),y_data,test_size = 0.2,random_state = 0)
X_train,X_valid,y_train,y_valid = train_test_split(x_data.to_numpy(),y_data,test_size = 0.1,random_state = 0)
model = XBNETClassifier(X_train,y_train,1)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

m,acc, lo, val_ac, val_lo = run_XBNET(X_train,X_valid,y_train,y_valid,model,criterion,optimizer,32,3)
filename = 'xbnet_model.pkl'
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
