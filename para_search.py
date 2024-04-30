import numpy as np
import matplotlib.pyplot as plt
import pickle
from data_loader import DataLoader
from network import Network
from train import train,evaluate

def parameter_search(X_train, y_train, X_val, y_val, grid_search,epoch):
    """
    参数查找
    """
    best_accuracy = 0
    best_parameters = None
    best_model = None
    # 网格筛选
    learning_rates = grid_search['lr']
    hidden_sizes = grid_search['hidden_size']
    hidden_sizes_2 = grid_search['hidden_size_2']
    reg_strengths = grid_search['reg']
    batch_size = grid_search['batch']
    param_result = {}
    for lr in learning_rates:
        for hidden_size in hidden_sizes:
            for hidden_size_2 in hidden_sizes_2:
                for batch in batch_size:
                    for reg_strength in reg_strengths:
                        print(f"learning_rate={lr}, hidden_size={hidden_size}, reg_strength={reg_strength},batch_size:{batch}")
                        model = Network(input_size=X_train.shape[1], hidden_size=hidden_size,hidden_size_2=hidden_size_2,output_size=10,learningrate=lr,
                                              activation='tanh',reg_strength=reg_strength)
                        parameters = train(model,X_train, y_train, X_val, y_val,
                                                 num_epochs=epoch, batch_size=batch)
                        x_pred = model.forward(X_val)
                        accuracy = evaluate(x_pred, y_val)
                        print(f"Validation Accuracy: {round(accuracy,3)}")
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_parameters = [lr,hidden_size,reg_strength,accuracy]
                            best_model = model
                        param_result[f'learning_rate={lr}, hidden_size={hidden_size},hidden_size_2 = {hidden_size_2}, reg_strength={reg_strength},batch_size:{batch}'] = accuracy
    # 存储最优模型及权重
    weights = (best_model.W1,best_model.W2,best_model.W3,best_model.A1,best_model.A2,best_model.A3)
    with open(fr'model/mybest_model_weights.pkl','wb') as f:
        pickle.dump(weights, f)
    with open(fr'model/mybest_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    return best_parameters

if __name__ == '__main__':
    dataloader = DataLoader('data/')
    X_train, y_train, X_val, y_val, X_test, y_test = dataloader.load_data()
    grid_search = {
    'lr': [0.005,0.01],
    'hidden_size':[64, 128, 256],
    'hidden_size_2':[64, 128],
    'reg' :[0.001, 0.01],
    'batch' :[32,64,128,256]
    }
    best_para = parameter_search(X_train, y_train, X_val, y_val, grid_search,epoch=20)