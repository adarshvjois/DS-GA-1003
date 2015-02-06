import pandas as pd
import logging
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import time
from mpl_toolkits.mplot3d.axes3d import Axes3D




def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.
    
    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test  - test set, a 2D numpy array of size (num_instances, num_features)
    Returns:
        train_normalized - training set after normalization
        test_normalized  - test set after normalization

    """
    # TODO
    for i in range(1, train.shape[1]):
        if train[: , i].min() == train[:, i].max() != 0:
            train[:, i] *= (1.0 / train[:, i].max())
            test[:, i] = test[:, i] * (1.0 / train[:, i].max())
            
        if train[:, i].min() != train[:, i].max():
            train[:, i] = (train[:, i] - train[:, i].min()) / (train[:, i].max() - train[:, i].min())
            test[:, i] = (test[:, i] - train[:, i].min()) / (train[:, i].max() - train[:, i].min())

    return train, test
    

    
########################################
####Q2.2a: The square loss function

def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the square loss for predicting y with X*theta
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)
    
    Returns:
        loss - the square loss, scalar
    """
    
    loss = np.dot(X, theta) - y
    return 0.5 * np.sum(loss ** 2) / X.shape[0]


########################################
# ##Q2.2b: compute the gradient of square loss function
def compute_square_loss_gradient(X, y, theta):
    """
    Compute gradient of the square loss (as defined in compute_square_loss), at the point theta.
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
    
    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    # TODO
    num_instances = X.shape[0]
    loss = np.dot(X, theta) - y
    theta = np.dot(X.T, loss)
    return (1.0 / num_instances) * theta
       
        
###########################################
# ##Q2.3a: Gradient Checker
# Getting the gradient calculation correct is often the trickiest part
# of any gradient-based optimization algorithm.  Fortunately, it's very
# easy to check that the gradient calculation is correct using the
# definition of gradient.
# See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4): 
    """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions: 
    (e_1 = (1,0,0,...,0), e_2 = (0,1,0,...,0), ..., e_d = (0,...,0,1) 

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by: 
    ( J(theta + epsilon * e_i) - J(theta - epsilon * e_i) ) / (2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error
    
    Return:
        A boolean value indicate whether the gradient is correct or not

    """
    true_gradient = compute_square_loss_gradient(X, y, theta)  # the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features)  # Initialize the gradient we approximate
    # TODO
    for i in range(num_features):
        
        theta_plus = theta.copy();
        theta_plus[i] = theta_plus[i] + epsilon
        theta_minus = theta.copy()
        theta_minus[i] = theta_minus[i] - epsilon
        approx_grad[i] = compute_square_loss(X, y, theta_plus) - compute_square_loss(X, y, theta_minus)
        approx_grad[i] /= (2.0 * epsilon)
    
    return np.sqrt(np.sum((true_gradient - approx_grad) ** 2)) < tolerance
    
#################################################
# ##Q2.3b: Generic Gradient Checker
def generic_gradient_checker(X, y, theta, objective_func, gradient_func, epsilon=0.01, tolerance=1e-4):
    """
    The functions takes objective_func and gradient_func as parameters. And check whether gradient_func(X, y, theta) returned
    the true gradient for objective_func(X, y, theta).
    Eg: In LSR, the objective_func = compute_square_loss, and gradient_func = compute_square_loss_gradient
    """
    # TODO
    true_gradient = compute_square_loss_gradient(X, y, theta)  # the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features)  # Initialize the gradient we approximate
    # TODO
    for i in range(num_features):
        
        theta_plus = theta.copy();
        theta_plus[i] = theta_plus[i] + epsilon
        theta_minus = theta.copy()
        theta_minus[i] = theta_minus[i] - epsilon
        approx_grad[i] = compute_square_loss(X, y, theta_plus) - compute_square_loss(X, y, theta_minus)
        approx_grad[i] /= (2.0 * epsilon)
    
    return np.sqrt(np.sum((true_gradient - approx_grad) ** 2)) < tolerance
        

####################################
####Q2.4a: Batch Gradient Descent
def batch_grad_descent(X, y, alpha=0.1, num_iter=1000, check_gradient=False):
    """
    In this question you will implement batch gradient descent to
    minimize the square loss objective
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        num_iter - number of iterations to run 
        check_gradient - a boolean value indicating whether checking the gradient when updating
        
    Returns:
        theta_hist - store the the history of parameter vector in iteration, 2D numpy array of size (num_iter+1, num_features) 
                    for instance, theta in iteration 0 should be theta_hist[0], theta in ieration (num_iter) is theta_hist[-1]
        loss_hist - the history of objective function vector, 1D numpy array of size (num_iter+1) 
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    loss_hist = np.zeros(num_iter + 1)  # initialize loss_hist
    theta = np.ones(num_features)  # initialize theta
    # TODO
    
    t0 = time.time()

    for i in xrange(num_iter + 1):
        
        loss_hist[i] = compute_square_loss(X, y, theta)
        theta_hist[i] = theta
        grad = compute_square_loss_gradient(X, y, theta) 
               
        if(grad_checker == True):
            if(grad_checker(X, y, theta)):
                logging.warn("Gradient estimated seems to deviate too far from what it should be")
            else:
                logging.info("Life's good.")

        theta = theta - alpha * grad
    t1 = time.time()

    print "Average time per iter:: " + str((t1 - t0) / num_iter)
    return theta_hist, loss_hist

def convergence_tests(X, y):
    num_iter = 500
    
    alpha = np.array([0.0005,0.005,0.001,0.01])
    loss_vs_alpha = np.zeros(len(alpha))
    plt.subplot(122)
    for i in range(len(alpha)):
        [theta, losses] = batch_grad_descent(X, y, alpha[i],num_iter)
        plt.plot(np.log(losses), label="Alpha(Batch Grad)" + str(alpha[i]))
    plt.legend()
####################################
# ##Q2.4b: Implement backtracking line search in batch_gradient_descent
# ##Check http://en.wikipedia.org/wiki/Backtracking_line_search for details
# TODO
    


###################################################
# ##Q2.5a: Compute the gradient of Regularized Batch Gradient Descent
def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    Compute the gradient of L2-regularized square loss function given X, y and theta
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        lambda_reg - the regularization coefficient
    
    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    # TODO
    regularization_term = 2.0 * lambda_reg * theta
    
    grad = compute_square_loss_gradient(X, y, theta) + regularization_term
    return grad
    

###################################################
# ##Q2.5b: Batch Gradient Descent with regularization term
def regularized_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000):
    """
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        numIter - number of iterations to run 
        
    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_iter+1, num_features) 
        loss_hist - the history of regularized loss value, 1D numpy array
    """
    (num_instances, num_features) = X.shape
    theta = np.ones(num_features)  # Initialize theta
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    loss_hist = np.zeros(num_iter + 1)  # Initialize loss_hist
    # TODO
    t0 = time.time()

    for i in range(num_iter + 1):
        loss_hist[i] = compute_square_loss(X, y, theta) + lambda_reg * np.sum(theta ** 2)
        theta_hist[i] = theta

        grad = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)
        theta = theta - alpha * grad
    
    t1 = time.time()
    #print "Average time per step:: "+str((t1 - t0)/num_iter) 
    return theta_hist, loss_hist

    
#############################################
# #Q2.5c: Visualization of Regularized Batch Gradient Descent
# #X-axis: log(lambda_reg)
# #Y-axis: square_loss
def visualize_loss_vs_lambda(X_train, X_test, y_train, y_test):
    
    X_train[:, -1] = np.ones(X_train.shape[0])
    X_train[:, -1] = 1 * X_train[:, -1]
    X_test[:, -1] = np.ones(X_test.shape[0])
    X_test[:, -1] = 1 * X_test[:, -1]
    
    lambdas = np.arange(1, 8 , 0.25)
    train_error = np.zeros(lambdas.shape[0])
    validation_error = np.zeros(lambdas.shape[0])
    i = 0
    for l in lambdas:
        [thetas, losses] = regularized_grad_descent(X_train, y_train, alpha=0.0001 *2.0/ 3, lambda_reg=l, num_iter=10000)
        train_error[i] = compute_square_loss(X_train, y_train, thetas[-1])
        validation_error[i] = compute_square_loss(X_test, y_test, thetas[-1])
        i += 1
    print "Min lambda:: " + str(lambdas[np.argmin(validation_error)])
    print "Min validation error::" + str(validation_error.min())
    print "Min training error:: " + str(train_error.min())
    plt.plot(np.log(lambdas), train_error,label="Training error", c='b')
    plt.plot(np.log(lambdas), validation_error,label="Validation Error" ,c='r')
    plt.legend()
    plt.xlabel("Lambda (log)")
    plt.ylabel("Error")
    plt.show()

def parameter_search(X_train, X_test, y_train, y_test):
    lambdas = 2.0 ** np.arange(-2, 8, 0.5)
    B = np.arange(1, 5.5, 0.5)
    train_error = np.zeros((B.shape[0], lambdas.shape[0]))
    validation_error = np.zeros((B.shape[0], lambdas.shape[0]))
    num_iter = 10000
    i = 0
    j = 0
    
    
    
    t0 = time.time()
    for lambda_reg in lambdas:
        j = 0    
        for b in B:
            X_train[:, -1] = np.ones(X_train.shape[0])
            X_train[:, -1] = b * X_train[:, -1]
            X_test[:, -1] = np.ones(X_test.shape[0])
            X_test[:, -1] = b * X_test[:, -1]
            [thetas, losses] = regularized_grad_descent(X_train, y_train,0.0001 * 2.0/3, lambda_reg, num_iter=num_iter)
            
            theta_best = thetas[-1]
            train_error[j, i] = compute_square_loss(X_train, y_train, theta_best)
            validation_error[j, i] = compute_square_loss(X_test, y_test, theta_best)
            j += 1
        i += 1
    t1 = time.time()
    
    print "Time to execute::" + str(t1 - t0)
    A = np.unravel_index(np.argmin(validation_error), validation_error.shape)
    print "Minimum cv error:: " + str(validation_error.min())
    print "Minimum train error:: " + str(train_error.min()) 
    print "Value of B, Lambda at lowest cross validation error: " + str(B[A[0]]) + " , " + str(lambdas[A[1]])
    
    # plt.plot(np.log(lambdas),train_error[:,0],'k',np.log(lambdas),validation_error[:,0],'r')
    (L, B_S) = np.meshgrid(lambdas, B)
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(L, B_S, train_error, rstride=1, cstride=1, cmap='cool', linewidth=0, antialiased=False, alpha=0.25)
    ax.set_xlabel('lambda (Log)')
    ax.set_ylabel('B ')
    ax.set_zlabel('train error)')
    
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(L, B_S, validation_error, rstride=1, cstride=1, cmap='cool', linewidth=0, antialiased=False, alpha=0.25)
    ax.set_xlabel('lambda (Log)')
    ax.set_ylabel('B ')
    ax.set_zlabel('validation error)')
    
    plt.show()


#############################################
# ##Q2.6a: Stochastic Gradient Descent
def stochastic_grad_descent(X, y, alpha=0.001, lambda_reg=1, num_iter=1000):
    """
    In this question you will implement stochastic gradient descent with a regularization term
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - string or float. step size in gradient descent
                NOTE: In SGD, it's not always a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every iteration is alpha.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t)
                if alpha == "1/t", alpha = 1/t
        lambda_reg - the regularization coefficient
        num_iter - number of epochs (i.e number of times) to go through the whole training set
    
    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size (num_iter, num_instances, num_features) 
        loss hist - the history of regularized loss function vector, 2D numpy array of size(num_iter, num_instances)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features)  # Initialize theta
    
    
    theta_hist = np.zeros((num_iter, num_instances, num_features))  # Initialize theta_hist
    loss_hist = np.zeros((num_iter, num_instances))  # Initialize loss_hist
    # TODO
    if isinstance(alpha, str):
        if alpha == '1/t':
            f = lambda x: 1.0 / x
        elif alpha == '1/sqrt(t)':
            f = lambda x: 1.0 / np.sqrt(x)
        alpha = 0.01
    elif isinstance(alpha, float):
        f = lambda x: 1
    else:
        return

    t0 = time.time()

    for t in range(num_iter):
        
        
        for i in range(num_instances):
            gamma_t = alpha * f((i+1)*(t+1))

            theta_hist[t , i] = theta
            # compute loss for current theta
            loss = np.dot(X[i], theta) - y[i]
            # reg. term
            regulariztion_loss = lambda_reg * np.dot(theta.T,theta)
            # squared loss
            loss_hist[t, i] = (0.5) * (loss) ** 2 + regulariztion_loss 
            
            regularization_penalty = 2.0 * lambda_reg * theta 
            grad = X[i] * (loss) + regularization_penalty
            theta = theta - gamma_t * grad
                        
    t1 = time.time()
    print "Average time per epoch:: " + str((t1 - t0) / num_iter) 
    return theta_hist, loss_hist

################################################
# ##Q2.6b Visualization that compares the convergence speed of batch
# ##and stochastic gradient descent for various approaches to step_size
# #X-axis: Step number (for gradient descent) or Epoch (for SGD)
# #Y-axis: log(objective_function_value)
def convergence_tests_batch_vs_stochastic(X,y):
    alphas = ['1/t','1/sqrt(t)',0.0005,0.001]
    plt.subplot(121)

    for alpha in alphas:
        [thetas,losses] = stochastic_grad_descent(X, y, alpha, 5.67, 5)
        plt.plot(np.log(losses.ravel()),label='Alpha:'+str(alpha))
        
    plt.legend()
    convergence_tests(X, y)
    plt.show()
    
def main():
    # Loading the dataset
    print('loading the dataset')
    
    df = pd.read_csv('hw1-data.csv', delimiter=',')
    X = df.values[:, :-1]
    y = df.values[:, -1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))  # Add bias term
    
    visualize_loss_vs_lambda(X_train, X_test, y_train, y_test)
    parameter_search(X_train, X_test, y_train, y_test)
    convergence_tests_batch_vs_stochastic(X_train,y_train)
    convergence_tests(X_train, y_train)
    plt.show() 
    
if __name__ == "__main__":
    main()
    
