import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x,w,b):
    z = x @ w + b
    return 1/( 1+np.exp(-z))

def log_likelihood(x,y,w,b):

    p = sigmoid(x,w,b)
    #p는 모든 x,y에 대한 확률이 나온다
    #for log clipping
    eps = 1e-12
    p = np.clip(p,eps,1-eps)

    #모든 p값에 대해서 log_likelyhood를 구한다

    return np.sum(y*np.log(p) + (1-y)*np.log(1-p))

def crossEntropy(x,y,w,b):

    N = x.shape[0]

    return -log_likelihood(x,y,w,b)/N

def grad_cross_entropy(x, y, w, b):
    """배치 평균 cross entropy의 gradient (loss를 줄이는 방향의 반대, 즉 descent용)"""
    p = sigmoid(x, w, b)
    dw = x.T @ (p - y) / x.shape[0]
    db = np.mean(p - y)
    return dw, db


def logisticRegression(x,y,learning_rate,epoches,batch_size,seed = None):
    

    N,D = x.shape
    
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(D)
    b = rng.standard_normal()

    history = {'w': [w.copy()], 'b': [b], 'loss': [crossEntropy(x, y, w, b)]}


    for epoch in range(epoches):

        idx = rng.permutation(N)

        for start in range(0,N, batch_size):
            batch = idx[start : start + batch_size]

            dw, db = grad_cross_entropy(x[batch], y[batch], w, b)

            w = w - learning_rate * dw
            b = b - learning_rate * db


        loss = crossEntropy(x,y,w,b)
        history['w'].append(w.copy())
        history['b'].append(b)
        history['loss'].append(loss)

        print(f"Epoch {epoch+1}/{epoches}, Loss: {loss:.4f}, "
        f"w: {np.round(w, 3)}, b: {b:.4f}")

    plt.plot(history['loss'])
    plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.show()
    return history
