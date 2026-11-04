import numpy as np
import matplotlib.pyplot as plt

def linear_f(x,w,b):
    #Multivariate linear function
    #return vector of y predict
    #[y1,y2,y4] = #[x1w1 +b ,x2w2+b,x3w3+b..]

    return x @ w + b

def mse_loss(x,y,w,b):
    #loss per sample
    #return loss vector
    y_pred = linear_f(x,w,b)
    return 0.5 * (y - y_pred) ** 2

def grad_w_mse(x,y,w,b):

    return (linear_f(x,w,b)-y) * x

def grad_b_mse(x,y,w,b):

    return (linear_f(x,w,b)-y) * 1


def Decay_Gradient_decent(x,y,learning_rate,batch_size,epochs,seed = None,grad_reduce="mean",decay=0.9):
    """
    Gradient descent algorithm to optimize the weight and bias
    :param x: input data
    :param y: true labels
    :param learning_rate: learning rate for gradient descent
    :param batch_size: size of each batch for mini-batch gradient descent
    :param epochs: number of iterations for gradient descent
    :return: optimized weight and bias
    """

    #N  is number of data, D is dimention of x
    N,D = x.shape
    
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(D)
    b = rng.standard_normal()

    history = {'w': [w], 'b': [b], 'loss': [np.mean(mse_loss(x,y,w,b))]}

    points = 0

    for epoch in range(epochs):
        #suffle data with idx
        idx = rng.permutation(N)
        #idx is array of suffle number between 0 ~ N 
        
        for start in range(0,N,batch_size):
            #from 0 ~ N step of batch_size
            
            batch = idx[start:start+batch_size]

            #dw => 0 vector sizeOF D
            #db = 0
            dw,db = np.zeros(D),0.0

            for j in batch :
                dw = dw + grad_w_mse(x[j],y[j],w,b)
                db = db + grad_b_mse(x[j],y[j],w,b)
            if grad_reduce == "mean":
                dw /= len(batch)
                db /= len(batch)

            w = w - learning_rate*dw
            b = b - learning_rate*db

            loss = np.mean(mse_loss(x,y,w,b))
            history['w'].append(w.copy())
            history['b'].append(b)
            history['loss'].append(loss)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, "
            f"w: {np.round(w, 3)}, b: {b:.4f}")

        learning_rate = learning_rate*decay


    plt.plot(history['loss'])
    plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.show()
    return history


def Decay_Gradient_decent2(x,y,learning_rate,batch_size,epochs,seed = None,grad_reduce="mean",decay=0.9):
    """
    Gradient descent algorithm to optimize the weight and bias
    :param x: input data
    :param y: true labels
    :param learning_rate: learning rate for gradient descent
    :param batch_size: size of each batch for mini-batch gradient descent
    :param epochs: number of iterations for gradient descent
    :return: optimized weight and bias
    """

    #N  is number of data, D is dimention of x
    N,D = x.shape
    
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(D)
    b = rng.standard_normal()

    history = {'w': [w], 'b': [b], 'loss': [np.mean(mse_loss(x,y,w,b))]}

    points = 0

    for epoch in range(epochs):
        #suffle data with idx
        idx = rng.permutation(N)
        #idx is array of suffle number between 0 ~ N 
        
        for start in range(0,N,batch_size):
            #from 0 ~ N step of batch_size
            
            batch = idx[start:start+batch_size]

            #dw => 0 vector sizeOF D
            #db = 0
            dw,db = np.zeros(D),0.0

            for j in batch :
                dw = dw + grad_w_mse(x[j],y[j],w,b)
                db = db + grad_b_mse(x[j],y[j],w,b)
            if grad_reduce == "mean":
                dw /= len(batch)
                db /= len(batch)

            w = w - learning_rate*dw
            b = b - learning_rate*db

            loss = np.mean(mse_loss(x,y,w,b))
            history['w'].append(w.copy())
            history['b'].append(b)
            history['loss'].append(loss)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, "
            f"w: {np.round(w, 3)}, b: {b:.4f}")
            
        learning_rate = learning_rate*decay
        decay = decay - 0.001


    plt.plot(history['loss'])
    plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.show()
    return history


