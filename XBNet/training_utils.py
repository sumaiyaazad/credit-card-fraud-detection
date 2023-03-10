import numpy as np
from sklearn.metrics import classification_report,r2_score,mean_absolute_error,mean_squared_error,mean_squared_log_error
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import csv,os
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay,accuracy_score,f1_score,precision_score,recall_score
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc, plot_precision_recall_curve
def training(model,trainDataload,testDataload,criterion,optimizer,epochs = 100,save = True):
    '''
    Training function for training the model with the given data
    :param model(XBNET Classifier/Regressor): model to be trained
    :param trainDataload(object of DataLoader): DataLoader with training data
    :param testDataload(object of DataLoader): DataLoader with testing data
    :param criterion(object of loss function): Loss function to be used for training
    :param optimizer(object of Optimizer): Optimizer used for training
    :param epochs(int,optional): Number of epochs for training the model. Default value: 100
    :return:
    list of training accuracy, training loss, testing accuracy, testing loss for all the epochs
    '''
    accuracy = []
    lossing = []
    val_acc = []
    val_loss = []
    header = ['epoch','Training accuracy','Training Loss','Validation accuracy','Validation Loss']
    p = model.get_params()
    my_path = format(os.getcwd())
    my_path = os.path.join(my_path,'result/xbnet')
    my_file = f'accuracy_tarining_layer='+str(p[0])+'_'+str(p[1])+'.csv'
    filename = os.path.join(my_path,my_file)
    
    file = open(filename,'w')
    writer=csv.writer(file,lineterminator='\n')
    writer.writerow(header)
    for epochs in tqdm(range(epochs),desc="Percentage training completed: "):
        running_loss = 0
        predictions = []
        act = []
        correct = 0
        total = 0
        loss = None
        for inp, out in trainDataload:
            try:
                if out.shape[0] >= 1:
                    out = torch.squeeze(out, 1)
            except:
                pass
            model.get(out.float())
            y_pred = model(inp.float())
            if model.labels == 1:
                loss = criterion(y_pred, out.view(-1, 1).float())
            else:
                loss = criterion(y_pred, out.long())
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            for i, p in enumerate(model.parameters()):
                if i < model.num_layers_boosted:
                    l0 = torch.unsqueeze(model.sequential.boosted_layers[i], 1)
                    lMin = torch.min(p.grad)
                    lPower = torch.log(torch.abs(lMin))
                    if lMin != 0:
                        l0 = l0 * 10 ** lPower
                        p.grad += l0
                    else:
                        pass
                else:
                    pass
            outputs = model(inp.float(),train = False)
            predicted = outputs
            total += out.float().size(0)
            if model.name == "Regression":
                pass
            else:
                if model.labels == 1:
                    for i in range(len(predicted)):
                        if predicted[i] < torch.Tensor([0.5]):
                            predicted[i] = 0
                        else:
                            predicted[i] =1

                        if predicted[i].type(torch.LongTensor) == out[i]:
                            correct += 1
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == out.long()).sum().item()

            predictions.extend(predicted.detach().numpy())
            act.extend(out.detach().numpy())
        lossing.append(running_loss/len(trainDataload))
        if model.name == "Classification":
            accuracy.append(100 * correct / total)
            #print("Training Loss after epoch {} is {} and Accuracy is {}".format(epochs + 1,
            #                                                                     running_loss / len(trainDataload),
            #                                                                     100 * correct / total))
        else:
            accuracy.append(100*r2_score(out.detach().numpy(),predicted.detach().numpy()))
            #print("Training Loss after epoch {} is {} and Accuracy is {}".format(epochs+1,running_loss/len(trainDataload),accuracy[-1]))
        v_l,v_a = validate(model,testDataload,criterion,epochs)
        
        
        print(f'epoch : {epochs+1}')
        print(f'Training accuracy   : {accuracy[-1]:.4f}  |  Training Loss : {lossing[-1]:.4f}')
        print(f'validation accuracy : {v_a[-1]:.4f}  |  Validation Loss : {float(v_l[-1]):.4f}')
        val_acc.extend(v_a)
        val_loss.extend(v_l)
        
        row = [epochs+1 , accuracy[-1],lossing[-1],v_a[-1],float(v_l[-1])]
        writer.writerow(row)
    if model.name == "Classification":
        print(classification_report(np.array(act),np.array(predictions)))
    else:
        print("R_2 Score: ", r2_score(np.array(act),np.array(predictions)))
        print("Mean Absolute error Score: ", mean_absolute_error(np.array(act),np.array(predictions)))
        print("Mean Squared error Score: ", mean_squared_error(np.array(act),np.array(predictions)))
        print("Root Mean Squared error Score: ", np.sqrt(mean_squared_error(np.array(act),np.array(predictions))))
    #validate(model,testDataload,criterion,epochs,True)

    #model.feature_importances_ = torch.nn.Softmax(dim=0)(model.layers["0"].weight[1]).detach().numpy()
    
    figure, axis = plt.subplots(2)
    figure.suptitle('Performance of XBNET')
    

    axis[0].plot(accuracy, label="Training Accuracy")
    axis[0].plot(val_acc, label="Testing Accuracy")
    axis[0].set_xlabel('Epochs')
    axis[0].set_ylabel('Accuracy')
    axis[0].set_title("XBNet Accuracy ")
    axis[0].legend()


    axis[1].plot(lossing, label="Training Loss")
    axis[1].plot(val_loss, label="Testing Loss")
    axis[1].set_xlabel('Epochs')
    axis[1].set_ylabel('Loss value')
    axis[1].set_title("XBNet Loss ")
    axis[1].legend()
    # Figures out the absolute path for you in case your working directory moves around.
    p= model.get_params()
    my_file = f'Training_graph_layer='+str(p[0])+'_'+str(p[1])+'.png'
    filename = os.path.join(my_path,my_file)
    print(filename,'-----------fpng')
    plt.savefig(filename)
    """if save == True:
        
    else:
        plt.show()"""
    
    
    return accuracy,lossing,val_acc,val_loss


@torch.no_grad()
def validate(model,testDataload,criterion,epochs,last=False):
    '''
    Function for validating the training on testing/validation data.
    :param model(XBNET Classifier/Regressor): model to be trained
    :param testDataload(object of DataLoader): DataLoader with testing data
    :param criterion(object of loss function): Loss function to be used for training
    :param epochs(int,optional): Number of epochs for training the model. Default value: 100
    :param last(Boolean, optional): Checks if the current epoch is the last epoch. Default: False
    :return:
    list of validation loss,accuracy
    '''
    valid_loss = 0
    accuracy = []
    lossing = []
    predictions = []
    act = []
    correct = 0
    total = 0
    
    
    for inp, out in testDataload:
        model.get(out.float())
        y_pred = model(inp.float(), train=False)
        
        if model.labels == 1:
            loss = criterion(y_pred, out.view(-1, 1).float())
        else:
            loss = criterion(y_pred, out.long())
        valid_loss += loss
        total += out.float().size(0)
        predicted = y_pred
        if model.name == "Regression":
            pass
        else:
            if model.labels == 1:
                for i in range(len(y_pred)):
                    if y_pred[i] < torch.Tensor([0.5]):
                        y_pred[i] = 0
                    else:
                        y_pred[i] = 1
                    if y_pred[i].type(torch.LongTensor) == out[i]:
                        correct += 1
            else:
                _, predicted = torch.max(y_pred.data, 1)
                correct += (predicted == out.long()).sum().item()

        predictions.extend(predicted.detach().numpy())
        act.extend(out.detach().numpy())
    lossing.append(valid_loss / len(testDataload))
    
    if model.name == "Classification":
        accuracy.append(100 * correct / total)
    else:
        accuracy.append(100 * r2_score(np.array(act), np.array(predictions)))
    if last:
        if model.name == "Classification":
            print(classification_report(np.array(act), np.array(predictions)))
        else:
            print("R_2 Score: ", r2_score(np.array(act), np.array(predictions)))
            print("Mean Absolute error Score: ", mean_absolute_error(np.array(act), np.array(predictions)))
            print("Mean Squared error Score: ", mean_squared_error(np.array(act), np.array(predictions)))
            print("Root Mean Squared error Score: ", np.sqrt(mean_squared_error(np.array(act), np.array(predictions))))
    """if model.name == "Classification":
        print("Validation Loss after epoch {} is {} and Accuracy is {}".format(epochs+1, valid_loss / len(testDataload),
                                                                               100 * correct / total))
        
                                                                              
    else:
        print("Validation Loss after epoch {} is {} and Accuracy is {}".format(epochs+1, valid_loss / len(testDataload),
                                                                                   100*r2_score(np.array(act), np.array(predictions))))
    """
    return lossing,   accuracy


def test(model,X,y):
    X = torch.from_numpy(X)
    y_pred = model(X.float(), train=False)
    
    y_test = torch.from_numpy(y)
    
    if model.name == "Classification":
        if model.labels == 1:
            if y_pred < torch.Tensor([0.5]):
                y_pred = 0
            else:
                y_pred = 1
        else:            
            y_pred = np.argmax(y_pred.detach().numpy(),axis=1)
        
    else:
        y_pred = y_pred.detach().numpy()[0]

    
    params = model.get_params()
    layers = params[0]
    last_layer = params[1]
    report =  classification_report(y_test,y_pred)

    print(report)
    my_path = format(os.getcwd())
    my_path = os.path.join(my_path,'result/xbnet')
    
    #calculate metrices
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


    my_file = f'classification_report_xbnet_test{params[0]}_{params[1]}.txt'
    filename = os.path.join(my_path,my_file)
    with open(filename,'w') as file:
        file.write(report)
        file.write(f'\nlayers: {layers}     |   last_layer : {last_layer}\n')
        file.write(f'\nTest accuracy : {acc}  |     f1-score(macro) : {f1_scr}\n')
        file.write(f'\nprecision     : {p}  |   recall  :  {r}\n ')
        file.write(f'\nArea under precision recall curve : {auc_precision_recall}')
    
    
    file = open('./result/test_result_xbnet.csv','a')
    writer=csv.writer(file,lineterminator='\n')
    
    row = [layers,last_layer,"{:.4f}".format(acc),"{:.4f}".format(p),"{:.4f}".format(r),"{:.4f}".format(f1_scr),"{:.4f}".format(auc_precision_recall)]
    writer.writerow(row)
    file.close()

    cm = confusion_matrix(y_test,y_pred)
    labels = [0,1]
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels =labels)
    cm_display.plot()
    plt.title(f'confusion matrix xbnet layer = {layers} {last_layer}')
    my_file = f'confusion_matrix_xbnet layer = {layers} {last_layer}.png'
    filename = os.path.join(my_path,my_file)
    plt.savefig(filename)




def predict(model,X):
    '''
    Predicts the output given the correct input data
    :param model(XBNET Classifier/Regressor): model to be trained
    :param X: Feature for which prediction is required
    :return:
    predicted value(int)
    '''
    X = torch.from_numpy(X)
    y_pred = model(X.float(), train=False)
    
    if model.name == "Classification":
        if model.labels == 1:
            if y_pred < torch.Tensor([0.5]):
                y_pred = 0
            else:
                y_pred = 1
        else:            
            y_pred = np.argmax(y_pred.detach().numpy(),axis=1)
        return y_pred
    else:
        return y_pred.detach().numpy()[0]

def predict_proba(model,X):
    '''
    Predicts the output given the correct input data
    :param model(XBNET Classifier/Regressor): model to be trained
    :param X: Feature for which prediction is required
    :return:
    predicted probabilties value(int)
    '''
    X = torch.from_numpy(X)
    y_pred = model(X.float(), train=False)
    return y_pred