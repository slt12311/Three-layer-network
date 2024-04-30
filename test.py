import numpy as np
from data_loader import DataLoader
from network import Network
import pickle
def evaluate(y_pred, y):
    # 准确率
    y_pred = np.argmax(y_pred, axis=1)
    accuracy = np.mean(y_pred == y)
    return accuracy

def test(X_test, y_test, model=False, model_weight = {}):
    # 导入模型
    if model==False:
        model = Network(input_size=784, hidden_size=128, hidden_size_2=128,output_size=10, learningrate=0.001,
                              activation='tanh',reg_strength=0.001)
        model.W1 = model_weight['W1']
        model.b1 = model_weight['b1']
        model.W2 = model_weight['W2']
        model.b2 = model_weight['b2']
        model.W3 = model_weight['W3']
        model.b3 = model_weight['b3']
    y_pred = model.forward(X_test)
    accuracy = evaluate(y_pred, y_test)
    return accuracy

if __name__ == '__main__':
    dataloader = DataLoader('data/')
    num_epochs,batch_size = 20,128
    X_train, y_train, X_val, y_val, X_test, y_test = dataloader.load_data()
    with open("model_weight.pkl", 'rb') as file:
        model_weight = pickle.loads(file.read())
    accu = test(X_test,y_test,model_weight=model_weight)
    print(f'Test Accuracy: {round(accu,3)}')
    with open("model.pkl", 'rb') as file:
        model = pickle.loads(file.read())
    accu = test(X_test,y_test,model=model)
    print(f'Test Accuracy: {round(accu,3)}')