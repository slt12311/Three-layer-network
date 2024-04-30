import numpy as np
import matplotlib.pyplot as plt
import pickle
from data_loader import DataLoader
from network import Network
def evaluate(y_pred, y):
    # 准确率
    y_pred = np.argmax(y_pred, axis=1)
    accuracy = np.mean(y_pred == y)
    return accuracy

def train(model,X_train, y_train, X_val, y_val, num_epochs=50, batch_size=32):
    # 网络训练
    best_val_accuracy = 0
    best_weight = None
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        epoch_train_loss = []

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # 前向传播
            y_pred = model.forward(X_batch)

            # 计算损失
            loss = model.cross_entropy_loss(y_pred, y_batch)
            # l2正则化
            m = len(y_batch)
            loss += 0.5/m * model.reg_strength * (np.sum(model.W1 ** 2) + np.sum(model.W2 ** 2)+ np.sum(model.W3 ** 2))
            epoch_train_loss.append(loss)

            # 反向传播
            model.backward(X_batch, y_batch, model.reg_strength)

        # 计算验证集上的损失和准确率
        y_pred = model.forward(X_val)
        val_loss = model.cross_entropy_loss(y_pred, y_val)
        val_accuracy = evaluate(y_pred, y_val)

        # 记录训练集上的损失和准确率
        train_losses.append(sum(epoch_train_loss)/len(epoch_train_loss))
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {round(val_accuracy,3)}")

        # 保存最佳模型参数
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_weight = {
                'W1': model.W1,
                'b1': model.b1,
                'W2': model.W2,
                'b2': model.b2,
                'W3': model.W3,
                'b3': model.b3
            }

    # 绘制损失曲线和准确率曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'hidden layer:{model.hidden_size},learning rate:{model.lr},batch size:{batch_size},reg:{model.reg_strength}')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'hidden layer:{model.hidden_size},learning rate:{model.lr},batch size:{batch_size},reg:{model.reg_strength}')

    plt.show()

    return best_weight

if __name__ == "__main__":
    dataloader = DataLoader('data/')
    X_train, y_train, X_val, y_val, X_test, y_test = dataloader.load_data()
    num_epochs, batch_size = 20, 32
    model = Network(input_size=X_train.shape[1], hidden_size=128, hidden_size_2=64,output_size=10, learningrate=0.01,
                    activation='tanh',reg_strength=0.01)
    model_weight = train(model, X_train, y_train, X_val, y_val, num_epochs, batch_size)
    with open(fr"model/model.pkl", "wb") as file:
        pickle.dump(model, file, True)
    with open(fr"model/model_weight.pkl","wb") as file:
        pickle.dump(model_weight, file, True)
