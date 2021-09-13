import numba 
import numpy as np
import time




'----------------------------------------------------------------------------------------------------------'
@numba.njit
def fi(i,x,y,w,ltype,lambda_):
    '''
    -------------------------------------------------------------------------------------------------------
    Computes the loss for a sample.
    -------------------------------------------------------------------------------------------------------
    Parameters:
        - i: int, the sample
        - x: np.array with shape (len(y),len(w)), dataset values
        - y: np.array with shape len(y), dataset labels
        - w: np.array with shape len(y), weights
        - ltype: int, the ltype of the loss function.

    Returns:
        - the loss calculated for the sample i
    -------------------------------------------------------------------------------------------------------
    '''
    xi = x[i]
    yi = y[i]

    lossi = 0

    if ltype == 1:
        lossi = np.log(1 + np.exp(-yi * np.sum(xi* w))) + lambda_/2 * np.sum(w*w)

    elif ltype == 2:
        lossi = np.log(1 + np.exp(-yi * np.sum(xi* w))) + lambda_ * np.sum( w**2 / (1 + w**2))

    elif ltype == 3:
        lossi = np.log(1 + 0.5 * (yi - np.sum(xi * w))**2)

    # else: 
    #     lossi = 0
    return lossi

'----------------------------------------------------------------------------------------------------------'



'----------------------------------------------------------------------------------------------------------'
@numba.njit
def grad_fi(i,x,y,w,ltype,lambda_):
    '''
    -------------------------------------------------------------------------------------------------------
    Computes the gradient for a sample.
    -------------------------------------------------------------------------------------------------------
    Parameters:
        - i: int, the sample
        - x: np.array with shape (len(y),len(w)), dataset values
        - y: np.array with shape len(y), dataset labels
        - w: np.array with shape len(y), weights
        - ltype: int, the ltype of the loss function. Can be 1,2 or 3.

    Returns:
        - the loss calculated for the sample i
    -------------------------------------------------------------------------------------------------------
    '''

    xi = x[i]
    yi = y[i]

    gradi = np.zeros(shape=np.shape(w))

    if ltype == 1:
        gradi = - yi * xi / (1 + np.exp(yi * np.sum(xi* w))) + lambda_ * w
    
    elif ltype == 2:
        gradi = - yi * xi / (1 + np.exp(yi * np.sum(xi* w))) + lambda_ * 2 * w/((1+w**2)**2)
    
    elif ltype == 3:
        gradi = - (yi - np.sum(xi*w)) * xi / (1 + 0.5 * (yi - np.sum(xi*w))**2)

    # else:
    #     gradi = np.zeros(shape=np.shape(w))
    return gradi
'----------------------------------------------------------------------------------------------------------'


'----------------------------------------------------------------------------------------------------------'
@numba.njit
def sum_gradients(sample,x,y,w,ltype,lambda_):
    '''
    -------------------------------------------------------------------------------------------------------
    For each integer in sample, sum the gradients of the specific sample and return that final sum
    -------------------------------------------------------------------------------------------------------
    Parameters:
        - sample: np.array with shape K, contains the integer of the samples of interest
        - x: np.array with shape (len(y),len(w)), dataset values
        - y: np.array with shape len(y), dataset labels
        - w: np.array with shape len(y), weight
        - ltype: int, the ltype of the loss function.
        - lambda_: float, the regularization parameter

    Returns:
        - the total gradient
    -------------------------------------------------------------------------------------------------------
    '''
    S = len(sample)
    grad = np.zeros(shape = np.shape(w))
    for i in sample:
        grad += grad_fi(i,x,y,w,ltype,lambda_)

    return grad/S
'----------------------------------------------------------------------------------------------------------'


'----------------------------------------------------------------------------------------------------------'
@numba.njit
def total_loss(x,y,weights,ltype,lambda_):
    '''
    -------------------------------------------------------------------------------------------------------
    Compute the total loss for the dataset, for each w in weights
    -------------------------------------------------------------------------------------------------------
    Parameters:
        - x: np.array with shape (len(y),len(w)), dataset values
        - y: np.array with shape len(y), dataset labels
        - weights: np.array, each element is the weight computed at each epoch
        - ltype: int, the ltype of the loss function.
        - lambda_: float, the regularization parameter

    Returns:
        - the loss for each epoch in weights
    -------------------------------------------------------------------------------------------------------
    '''
    losses = np.zeros(shape = len(weights))
    n = len(y)
    for k,w in enumerate(weights):
        loss = 0
        for i in range(n):
            loss += fi(i,x,y,w,ltype,lambda_)
        losses[k] = loss/len(y)

    return losses
'----------------------------------------------------------------------------------------------------------'


'----------------------------------------------------------------------------------------------------------'
@numba.njit
def Vanilla_GD(x,y,w0,lr,ltype,lambda_,n_epochs):
    '''
    -------------------------------------------------------------------------------------------------------
    Implementation of the Gradient Descnet algorithm
    -------------------------------------------------------------------------------------------------------
    Parameters:
        - x: np.array with shape (len(y),len(w)), dataset values
        - y: np.array with shape len(y), dataset labels
        - w0: np.array with shape len(y), the initial weight
        - lr: float, the learning rate
        - ltype: int, the ltype of the loss function.
        - lambda_: float, the regularization parameter
        - n_epochs: int, the number of epochs


    Returns:
        - the updated weight at each iteration
    -------------------------------------------------------------------------------------------------------
    '''
    w = w0 
    n = len(y)
    sample = np.arange(n)
    for _ in range(n_epochs):
        w = w - lr * sum_gradients(sample,x,y,w,ltype,lambda_)
        w_yield = w
        yield w_yield
'----------------------------------------------------------------------------------------------------------'


'----------------------------------------------------------------------------------------------------------'
@numba.njit
def Stochastic_GD(x,y,w0,lr,S,ltype,lambda_,n_epochs):
    '''
    -------------------------------------------------------------------------------------------------------
    Implementation of the Mini-batches Stochastic Gradient Descent algorithm
    -------------------------------------------------------------------------------------------------------
    Parameters:
        - x: np.array with shape (len(y),len(w)), dataset values
        - y: np.array with shape len(y), dataset labels
        - w0: np.array with shape len(y), the initial weight
        - lr: float, the learning rate
        - S: int, the size of the mini-batch
        - ltype: int, the ltype of the loss function.
        - lambda_: float, the regularization parameter
        - n_epochs: int, the number of epochs


    Returns:
        - the updated weight at each iteration
    -------------------------------------------------------------------------------------------------------
    '''
    w = w0 
    n = len(y)
    choose_from = np.arange(n)
    for _ in range(n_epochs):
        sample = np.random.choice(choose_from,size = S,replace=True)
        w = w - lr * sum_gradients(sample,x,y,w,ltype,lambda_)
        w_yield = w
        yield w_yield
'----------------------------------------------------------------------------------------------------------'


'----------------------------------------------------------------------------------------------------------'
@numba.njit
def SARAH(x,y,w0,lr,m,ltype,lambda_,n_epochs):
    '''
    -------------------------------------------------------------------------------------------------------
    Implementation of Sarah algorithm
    -------------------------------------------------------------------------------------------------------
    Parameters:
        - x: np.array with shape (len(y),len(w)), dataset values
        - y: np.array with shape len(y), dataset labels
        - w0: np.array with shape len(y), the initial weight
        - lr: float, the learning rate
        - m: size of the inner loop
        - ltype: int, the ltype of the loss function.
        - lambda_: float, the regularization parameter
        - n_epochs: int, the number of epochs


    Returns:
        - the updated weight at each iteration
    -------------------------------------------------------------------------------------------------------
    '''
    w = w0
    n = len(y)
    sample = np.arange(n)
    for _ in range(n_epochs):
        w_ = w 
        v = sum_gradients(sample,x,y,w,ltype,lambda_)
        w = w - lr*v
        for _ in range(m):
            i = np.random.randint(0,n)
            v = grad_fi(i,x,y,w,ltype,lambda_) - grad_fi(i,x,y,w_,ltype,lambda_) + v 
            w_ = w 
            w = w - lr * v
        w_yield = w 
        yield w_yield 
'----------------------------------------------------------------------------------------------------------'

'----------------------------------------------------------------------------------------------------------'
@numba.njit
def SARAH_plus(x,y,w0,lr,m,gamma,ltype,lambda_,n_epochs):
    '''
    -------------------------------------------------------------------------------------------------------
    Implementation of Sarah algorithm
    -------------------------------------------------------------------------------------------------------
    Parameters:
        - x: np.array with shape (len(y),len(w)), dataset values
        - y: np.array with shape len(y), dataset labels
        - w0: np.array with shape len(y), the initial weight
        - lr: float, the learning rate
        - m: int, the max iteration for the inner loop
        - gamma: float, the gamma parameter
        - ltype: int, the ltype of the loss function.
        - lambda_: float, the regularization parameter
        - n_epochs: int, the number of epochs


    Returns:
        - the updated weight at each iteration
    -------------------------------------------------------------------------------------------------------
    '''
    w = w0
    n = len(y)
    sample = np.arange(n)
    for _ in range(n_epochs):
        w_ = w 
        v0 = sum_gradients(sample,x,y,w,ltype,lambda_)
        w = w - lr*v0 

        v = v0
        t = 1 
        while t < m and np.sum(v*v) > gamma * np.sum(v0*v0):
            i = np.random.randint(0,n)
            v = grad_fi(i,x,y,w,ltype,lambda_) - grad_fi(i,x,y,w_,ltype,lambda_) + v 
            w_ = w 
            w = w - lr * v
            t+=1
        w_yield = w 
        yield w_yield 
'----------------------------------------------------------------------------------------------------------'

'----------------------------------------------------------------------------------------------------------'
@numba.njit
def spiderboost_vk(sample,x,y,w,w_,ltype,lambda_):
    '''
    -------------------------------------------------------------------------------------------------------
    Helper for the Spider ltypes of algorithms.
    -------------------------------------------------------------------------------------------------------
    Parameters:
    - x: np.array with shape (len(y),len(w)), dataset values
    - y: np.array with shape len(y), dataset labels
    - w: np.array with shape len(y),the weight
    - w: np.array with shape len(y),the previous weight    
    - ltype: int, the ltype of the loss function.
    - lambda_: float, the regularization parameter
    -------------------------------------------------------------------------------------------------------
    '''

    vk = np.zeros(shape = np.shape(w))
    for i in sample:
        vk += grad_fi(i,x,y,w,ltype,lambda_) - grad_fi(i,x,y,w_,ltype,lambda_) 
    return vk/len(sample)
'----------------------------------------------------------------------------------------------------------'


'----------------------------------------------------------------------------------------------------------'
@numba.njit
def Spider(x,y,w0,lr,q,K,S,eps_tilde,ltype,lambda_,n_epochs):
    '''
    -------------------------------------------------------------------------------------------------------
    Implementation of Spider algorithm
    -------------------------------------------------------------------------------------------------------
    Parameters:
        - x: np.array with shape (len(y),len(w)), dataset values
        - y: np.array with shape len(y), dataset labels
        - w0: np.array with shape len(y), the initial weight
        - lr: float, the learning rate
        - q: int, compute full gradient every q interactions
        - K: int, the size of the inner loop
        - S: int, the size of the mini-batches
        - eps_tilde: float, it's the eps tilde of the algorithm
        - ltype: int, the ltype of the loss function.
        - lambda_: float, the regularization parameter
        - n_epochs: int, the number of epochs


    Returns:
        - the updated weight at each iteration
    -------------------------------------------------------------------------------------------------------
    '''
    w = w0 
    
    n = len(y) 
    full_sample = np.arange(n)
    for _ in range(n_epochs):
        w_ = w
        'inner loop'
        for k in range(K):
            if np.mod(k,q)==0:
                v = sum_gradients(full_sample,x,y,w,ltype,lambda_)
            else: 
                sample = np.random.choice(full_sample,size=S,replace=True)
                v = spiderboost_vk(sample,x,y,w,w_,ltype,lambda_) + v
            
            w_ = w
            normv  = np.linalg.norm(v)
            
            if normv <= 2*eps_tilde:
                w_yield = w 
                yield w_yield 
            else:
                w = w - lr * v/normv
        w_yield = w 
        yield w_yield
'----------------------------------------------------------------------------------------------------------'


'----------------------------------------------------------------------------------------------------------'
@numba.njit
def SpiderBoost(x,y,w0,lr,q,K,S,ltype,lambda_,n_epochs):
    '''
    -------------------------------------------------------------------------------------------------------
    Implementation of Spiderboost algorithm
    -------------------------------------------------------------------------------------------------------
    Parameters:
        - x: np.array with shape (len(y),len(w)), dataset values
        - y: np.array with shape len(y), dataset labels
        - w0: np.array with shape len(y), the initial weight
        - lr: float, the learning rate
        - q: int, compute full gradient every q interactions
        - K: int, the size of the inner loop
        - S: int, the size of the mini-batches
        - ltype: int, the ltype of the loss function.
        - lambda_: float, the regularization parameter
        - n_epochs: int, the number of epochs


    Returns:
        - the updated weight at each iteration
    -------------------------------------------------------------------------------------------------------
    '''
    w = w0 
    
    n = len(y) 
    full_sample = np.arange(n)
    for _ in range(n_epochs):
        w_ = w
        'inner loop'
        for k in range(K):
            if np.mod(k,q)==0:
                v = sum_gradients(full_sample,x,y,w,ltype,lambda_)
            else: 
                sample = np.random.choice(full_sample,size=S,replace=True)
                v = spiderboost_vk(sample,x,y,w,w_,ltype,lambda_) + v
            w_ = w
            w = w - lr * v 
        w_yield = w 
        yield w_yield
'----------------------------------------------------------------------------------------------------------'

'----------------------------------------------------------------------------------------------------------'
@numba.njit
def SpiderBoost_m(x,y,w0,lr,q,K,S,ltype,lambda_,n_epochs):
    '''
    -------------------------------------------------------------------------------------------------------
    Implementation of Spiderboost algorithm
    -------------------------------------------------------------------------------------------------------
    Parameters:
        - x: np.array with shape (len(y),len(w)), dataset values
        - y: np.array with shape len(y), dataset labels
        - w0: np.array with shape len(y), the initial weight
        - lr: float, the learning rate
        - q: int, compute full gradient every q interactions
        - K: int, the size of the inner loop
        - S: int, the size of the mini-batches
        - ltype: int, the ltype of the loss function.
        - lambda_: float, the regularization parameter
        - n_epochs: int, the number of epochs


    Returns:
        - the updated weight at each iteration
    -------------------------------------------------------------------------------------------------------
    '''
    w = w0 
    
    n = len(y) 
    full_sample = np.arange(n)

    beta_k = lr 
    for _ in range(n_epochs):
        yk = w 
        w_ = w
        'inner loop'
        for k in range(K):
            alpha_k = 2/((k/q)+1)
            alpha_k1 = 2/((k+1)/q + 1)

            lambda_k = (1+alpha_k)*beta_k

            zk = (1 - alpha_k1)*yk + alpha_k1 * w
            if np.mod(k,q)==0:
                v = sum_gradients(full_sample,x,y,w,ltype,lambda_)
            else: 
                sample = np.random.choice(full_sample,size=S,replace=True)
                v = spiderboost_vk(sample,x,y,w,w_,ltype,lambda_) + v
            w_ = w
            yk = zk - beta_k * v
            w = w - lambda_k * v


        z_yield = zk
        yield z_yield
'----------------------------------------------------------------------------------------------------------'

'----------------------------------------------------------------------------------------------------------'
@numba.njit
def SVRG(x,y,w0,lr,m,n_epochs,ltype,lambda_):
    '''
    -------------------------------------------------------------------------------------------------------
    Implementation of SVRG algorithm
    -------------------------------------------------------------------------------------------------------
    Parameters:
        - x: np.array with shape (len(y),len(w)), dataset values
        - y: np.array with shape len(y), dataset labels
        - w0: np.array with shape len(y), the initial weight
        - lr: float, the learning rate
        - m: int, the size of the inner loop
        - ltype: int, the ltype of the loss function.
        - lambda_: float, the regularization parameter
        - n_epochs: int, the number of epochs


    Returns:
        - the updated weight at each iteration
    -------------------------------------------------------------------------------------------------------
    '''
    w = w0 
    n = len(y)
    full_sample = np.arange(n)
    for _ in range(n_epochs):
        w_ = w 
        mu = sum_gradients(full_sample,x,y,w,ltype,lambda_)
        for _ in range(m):
            i = np.random.randint(0,n)
            w_ = w 
            w = w - lr * (grad_fi(i,x,y,w,ltype,lambda_) - grad_fi(i,x,y,w_,ltype,lambda_) + mu)
        
        w_yield = w
        yield w_yield
'----------------------------------------------------------------------------------------------------------'

'----------------------------------------------------------------------------------------------------------'
def wrapper(alg,x,y,w0,min_loss,eps,max_iter,wait,params,lambda_,ltype,return_times,return_weights):
    '''
    ---------------------------------------------------------------------------------------------------
    Wrapper built around the implemented algorithms. Specify an eps and an alg, and this function will keep 
    calling the alg untill P(w)-P(w*)<=eps.
    ---------------------------------------------------------------------------------------------------
    Parameters:
        - alg: string, the algorithm of interest. Can be:
            * gd;
            * sgd;
            * sarah;
            * sarahplus;
            * spider;
            * spiderboost;
            * spiderboost-m;
            * svrg.
        - y: np array, the dataset labels;
        - x: np array, the dataset values
        - w0: the np array of shape len(y), initial weight;
        - min_loss: float, the optimal loss value for the dataset;
        - eps: float, the threshold such that P(w)-P(w*)<=eps
        - max_iter: int, the max number of iterations;
        - wait: int, the number of iterations to wait before interrupting;
        - params: list, contains the parameters for the chosen algorithm;
        - lambda_: the regularization parameter of the loss;
        - ltype: the loss type (can be 1,2 or 3)
        - return_times: bool, if True return the losses and the times
        - return_weights: bool, if True return the losses and the weights
    
    Return:
        - the list of losses for each iteration, the times and the weights (if wanted)
    ---------------------------------------------------------------------------------------------------
    '''

    if alg == 'gd':
        this_alg = lambda w: Vanilla_GD(x=x,y=y,w0=w,lr=params[0],lambda_=lambda_,ltype=ltype, n_epochs=1)
    if alg == 'sgd':
        this_alg = lambda w: Stochastic_GD(x=x,y=y,w0=w,lr=params[0],S=params[1],lambda_=lambda_,ltype=ltype, n_epochs=1)
    if alg == 'sarah':
        this_alg = lambda w: SARAH(x=x,y=y,w0=w,lr=params[0],m=params[1],lambda_=lambda_,ltype=ltype, n_epochs=1)
    if alg == 'sarahplus':
        this_alg = lambda w: SARAH_plus(x=x,y=y,w0=w,lr=params[0], m=params[1],gamma=params[2],lambda_=lambda_,ltype=ltype, n_epochs=1)
    if alg == 'spider':
        this_alg = lambda w: Spider(x=x,y=y,w0=w,lr=params[0],K = params[1], q=params[2],S=params[3],eps_tilde = params[4],lambda_=lambda_,ltype=ltype, n_epochs=1)
    if alg == 'spiderboost':
        this_alg = lambda w: SpiderBoost(x=x,y=y,w0=w,lr=params[0],K = params[1], q=params[2],S=params[3],lambda_=lambda_,ltype=ltype, n_epochs=1)
    if alg == 'spiderboost-m':
        this_alg = lambda w: SpiderBoost_m(x=x,y=y,w0=w,lr=params[0],K = params[1], q=params[2],S=params[3],lambda_=lambda_,ltype=ltype, n_epochs=1)
    if alg == 'svrg':
        this_alg = lambda w: SVRG(x=x,y=y,w0=w,lr=params[0],m=params[1],ltype=ltype,lambda_=lambda_,n_epochs=1)
    count = 0 
    waiting = 0
    last_loss = np.inf
    losses = [total_loss(x,y,np.array([w0]),ltype,lambda_)[0]]
    times = [0]
    weights = [w0]
    w = w0
    while waiting < wait and count < max_iter:
        start = time.time()
        w = list(this_alg(w))[0]
        end = time.time()
        loss = total_loss(x,y,np.array([w]),ltype,lambda_)[0]
        losses.append(loss)
        weights.append(w)
        times.append(end-start) 
        
        count+=1
        if (np.abs(loss - min_loss) <= eps) or loss > last_loss:
            waiting += 1 

        last_loss = loss

    losses = np.asarray(losses)
    weights = np.asarray(weights)
    times = np.asarray(times)
    if not return_times and not return_weights:
        return losses
    elif return_times and not return_weights:
        return losses, times
    elif not return_times and return_weights:
        return losses, weights
    else:
        return losses, times, weights
'----------------------------------------------------------------------------------------------------------'


'----------------------------------------------------------------------------------------------------------'
# @numba.njit
def compute_errors(x,y,w):
    '''
    ----------------------------------------------------------------------------------------------------------
    Compute errors on the test set.
    ----------------------------------------------------------------------------------------------------------
    '''
    preds = 1/(1+np.exp(-np.sum(x*w,axis=-1)))
    preds = preds >= 0.5 
    preds = preds * 1
    preds = preds * 2 - 1

    return np.mean(preds==y)
'----------------------------------------------------------------------------------------------------------'