# 代码结构

data：存储下载的数据集

model：存储训练的最优模型权重

data_loader.py：数据读取和处理，定义DataLoader类

network.py：定义神经网络Network类

train.py：模型训练，定义train函数

test.py：模型测试和评估

para_search.py：模型调参，定义parameter_search函数

# 模型训练

在train.py函数中：

- 数据读取：调用DataLoader类及load_data()函数读入'data/'下数据
- 初始化神经网络：调用Network类定义神经网络模型，输入input_size, hidden_size, hidden_size_2,output_size, learningrate, activation,reg_strength参数取值，其中activation可选参数包括'sigmoid'/'tanh'/'relu'
- 模型训练：调用train函数训练模型，并逐轮打印验证集上准确率，训练完成后打印训练集和验证集上的loss曲线和验证集上的accuracy曲线
- 模型存储：采用pickle.dump()存储模型

# 模型测试

在test.py函数中：

- 数据读取：调用DataLoader类及load_data()函数读入'data/'下数据；也可自定义数据读取，定义X_test和y_test即可
- 模型读取：采用pickle.loads()读取训练好的模型
- 模型测试：调用test函数测试模型在测试集上的表现，并输出准确率

