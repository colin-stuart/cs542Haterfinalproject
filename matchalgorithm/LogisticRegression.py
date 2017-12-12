import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import sklearn
import seaborn as sn


# Sigmoid function

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

# ReLu function

def ReLU(x):
    return x * (x > 0)

# Hyperbolic tan

def tanh(x):
    return np.tanh(x)


def initialize_weight_and_bias(dimension):
    w = 1e-4 * np.random.randn(dimension, 1)
    b = 0
    return w, b

def forward_prop(X, w, b):
    return sigmoid(np.dot(X, w) + b)

def compute_cost(pred, Y):
    w, b = initialize_weight_and_bias(Y.shape[1])
    cost = (-1 / Y.shape[0]) * np.sum(Y * np.log(pred) + (1 - Y) * np.log(1 - pred))
    return cost

def back_prop(X, pred, Y):
    dw = (1 / Y.shape[0]) * np.dot((pred - Y).T, X)
    db = (1 / Y.shape[0]) * np.sum(pred - Y)
    return {"dw": dw, "db": db}

def optimize(w, b, X, Y, num_iterations, learning_rate):
    costs = []

    for i in range(num_iterations):

        pred = forward_prop(X, w, b)
        cost = compute_cost(pred, Y)
        grads = back_prop(X, pred, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw.T
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    m = X.shape[0]
    Y_prediction = np.zeros((1, m))

    pred = sigmoid(np.dot(X, w) + b)

    print("Printing predictions..")     # Debugging purposes
    print(pred)     # Debugging purposes

    for i in range(pred.shape[0]):

        if pred[i,0] <= 0.50:   # baseline threshold that indicates 50% chance of matching
             Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    return Y_prediction


def modified_model(X_train, Y_train, num_iterations=10000, learning_rate=0.005):

    w, b = initialize_weight_and_bias(X_train.shape[1])

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_train = predict(w, b, X_train)

#### PLOT FOR COSTS ####
    num_it = []
    for i in range(num_iterations):
        if i % 100 == 0:
            num_it += [i]

    plt.plot(num_it, costs)
    plt.title("Number of iterations vs costs")
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.show()

# COUNTING Actual vs Prediction

    # COUNT OF ACTUAL
    zero_actual = 0
    one_actual = 0
    for element in Y_train:
        if element == 0:
            zero_actual += 1
        else:
            one_actual += 1

    # COUNT OF PREDICTION
    zero_pred = 0
    one_pred = 0
    for element in Y_prediction_train[0]:
        if element == 0:
            zero_pred += 1
        else:
            one_pred += 1

    # Display this information
    print("Number of zeros in actual:", zero_actual)
    print("Number of ones in actual:", one_actual)
    print("Number of zeros in prediction:", zero_pred)
    print("Number of ones in prediction:", one_pred)

    # Prepare a confusion matrix for this data
    conf_mat = np.array([[zero_actual,one_pred], [zero_pred,one_actual]])

    # Plotting the confusion matrix
    title_font = {'fontname': 'Arial', 'size': '16', 'color': 'black', 'weight': 'normal',
                  'verticalalignment': 'bottom'}
    sn.set(font_scale=2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_mat, cmap="GnBu")
    plt.title('Matches Confusion Matrix', y=1.08)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + ["0", "1"])
    ax.set_yticklabels([''] + ["0", "1"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    #################################################

    # Calculating the train accuracy and displaying
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))

    # Recording information such as cost, prediction, weights, biases, and also learning rate and number of iterations
    d = {"costs": costs,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    return d


################# One model for both test and train ###############################

#     def model(X_train, Y_train, X_test, Y_test, num_iterations=10000, learning_rate=0.005):
#     w, b = initialize_weight_and_bias(X_train.shape[1])
#
#     parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)
#
#     w = parameters["w"]
#     b = parameters["b"]
#
#     Y_prediction_test = predict(w, b, X_test)
#     Y_prediction_train = predict(w, b, X_train)
#
#     print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
#     print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
#
#     d = {"costs": costs,
#          "Y_prediction_test": Y_prediction_test,
#          "Y_prediction_train": Y_prediction_train,
#          "w": w,
#          "b": b,
#          "learning_rate": learning_rate,
#          "num_iterations": num_iterations}
#
#     num_it = []
#     for i in range(num_iterations):
#         if i % 100 == 0:
#             num_it += [i]
#
#     plt.plot(num_it, costs)
#     plt.title("Number of iterations vs costs")
#     plt.xlabel("Number of iterations")
#     plt.ylabel("Cost")
#     plt.show()
#
#     return d
