import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


def softmax(Z):
    Z = Z - np.max(Z, axis=1, keepdims=True)
    E_z = np.exp(Z)
    return E_z / np.sum(E_z, axis=1, keepdims=True)


def train(X, labels, K, lr=0.1, epochs=500):
    m, i = X.shape
    W = np.zeros((K, i))
    b = np.zeros(K)
    Y = np.eye(K)[labels]       
    ##eye는 K X K 행렬이고 Lables에 있는 정수는 행마다 one hot
    #encoding의 위치를 출력 (m, K) one-hot

    history = []

    for ep in range(epochs):
        # forward (루프 안!)
        Z = X @ W.T + b            # (m, K)
        P = softmax(Z)             # (m, K)

        # loss
        loss = -np.sum(Y * np.log(P + 1e-12)) / m
        history.append(loss)

        # backward
        # backward is always same in GLM
        dW = (P - Y).T @ X / m     # (K, i)
        db = (P - Y).mean(axis=0)  # (K,)
        assert dW.shape == W.shape and db.shape == b.shape

        # update
        W = W - lr * dW
        b = b - lr * db

    return W, b, history

def train_batch(X,batch_size, labels, K, lr=0.1, epochs=500,seed = 42):

    m, i = X.shape
    W = np.zeros((K, i))
    b = np.zeros(K)
    Y = np.eye(K)[labels]
    rng = np.random.default_rng(seed)

    history = []

    for epoch in range(epochs):

        idx = rng.permutation(m)

        for start in range(0,m,batch_size):

            batch = idx[start:start+batch_size]

            Xb = X[batch] #(B,i)
            Yb = Y[batch] #(B,K)
            
            B= len(batch)

            P = softmax(Xb@W.T + b)             # (m, K)
            
            # backward
            # backward is always same in GLM
            dW = (P - Yb).T @ Xb / B     # (K, i)
            db = (P - Yb).mean(axis=0)  # (K,)
            assert dW.shape == W.shape and db.shape == b.shape

            #update
            W = W - lr * dW
            b = b - lr * db
            
        P_all = softmax(X @ W.T + b)
        history.append(-np.sum(Y * np.log(P_all + 1e-12)) / m)

    return W, b, history

def evaluate(X_test,labels_test,W,b):

    #X를 완성된 선형 모델과 시그모이드 함수에 집어 넣고 loss와 정답률 게산해서 출력

    m = X_test.shape[0]
    Z =  X_test @ W.T + b
    P = softmax(Z)

    #정답칸의 확률만 꺼내서 y와 곱 --> loss
    #정답이 아닌 건 0과 곱하기 때문
    loss = -np.mean(np.log(P[np.arange(m), labels_test] + 1e-12))
    
    pred = P.argmax(axis=1)                        # (m,)
    acc = (pred == labels_test).mean()             # True/False의 평균 = 정답 비율

    return loss, acc