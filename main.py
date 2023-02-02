import math
import numpy as np
import matplotlib.pyplot as plt
import copy

def whichCityToOpenANewFranchise():
    """
    Problem:
    Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet.

    -You would like to expand your business to cities that may give your restaurant higher profits.
    -The chain already has restaurants in various cities and you have data for profits and populations from the cities.
    -You also have data on cities that are candidates for a new restaurant.
        -For these cities, you have the city population.

    Can you use the data to help you identify which cities may potentially give your business higher profits?
    """

    file = open("./whichCityToOpenANewFranchiseData.txt","r")
    x_trainArray = []
    y_trainArray = []
    for aline in file:
        values = aline.split(",")
        x_trainArray.append(float(values[0]))
        y_trainArray.append(float(values[1]))

    # For x train set:
    # These values represent the city population times 10,000
    # For example, 6.1101 means that the population for that city is 61,101
    x_train = np.array(x_trainArray)

    # For y train set:
    # These represent your restaurant's average monthly profits in each city, in units of $10,000
    # For example, 17.592 represents $175,920 in average monthly profits for that city.
    # -2.6807 represents -$26,807 in average monthly loss for that city.
    y_train = np.array(y_trainArray)

    #Now let's check the first five values
    print("Type of x_train:", type(x_train))
    print("First five elements of x_train are:\n", x_train[:5])
    print("-----------------------------")


    # Now check the dimensions of numpy arrays
    print('The shape of x_train is:', x_train.shape)
    print('The shape of y_train is:', y_train.shape)
    print("Number of training examples (m): ", len(x_train))
    print("-----------------------------")


    # Let's visualise our data
    plt.scatter(x_train,y_train,marker="x",c="r")
    plt.title("Profits vs. Population per city")
    plt.xlabel("Population of city in 10.000s")
    plt.ylabel("Profit in $10.000")
    plt.show()
    print("-----------------------------")

    #Let's check our computeCost function with some initial variables
    initialW = 2
    initialB = 1
    cost = computeCost(x_train,y_train,initialW,initialB)
    print(type(cost))
    print(f'Cost at initial w: {cost:.3f}')
    print("-----------------------------")


    # Let's check our computeGradient function with some initial variables
    initialW = 0.2
    initialB = 0.2
    tmp_dj_dw, tmp_dj_db = computeGradient(x_train, y_train, initialW, initialB)
    print('Gradient at initial w, b (zeros):', tmp_dj_dw, tmp_dj_db)
    print("-----------------------------")

    # Let's check our gradientDescent function with some initial variables
    initialW = 0.
    initialB = 0.
    # some gradient descent settings
    iterations = 1500
    alpha = 0.01
    w, b, _, _ = gradientDescent(x_train, y_train, initialW, initialB,computeCost, computeGradient, alpha, iterations)
    print("w,b found by gradient descent:", w, b)
    print("-----------------------------")

    # To calculate the predictions on the entire dataset, we can loop through all the training
    #   examples and calculate the prediction for each example. This is shown in the code block below.
    m = x_train.shape[0]
    predicted = np.zeros(m)

    for i in range(m):
        predicted[i] = w * x_train[i] + b

    # Plot the linear fit
    plt.plot(x_train, predicted, c="b")

    # Create a scatter plot of the data.
    plt.scatter(x_train, y_train, marker='x', c='r')
    plt.title("Profits vs. Population per city")
    plt.ylabel('Profit in $10,000')
    plt.xlabel('Population of City in 10,000s')
    plt.show()
    print("-----------------------------")

    # Your final values of w,b can also be used to make predictions on profits. Let's predict what the profit
    #   would be in areas of 35.000 and 70.000 people
    predict1 = 3.5*w + b
    print('For population = 35,000, we predict a profit of $%.2f' % (predict1 * 10000))

    predict2 = 7.0*w + b
    print('For population = 70,000, we predict a profit of $%.2f' % (predict2 * 10000))

def computeCost(x,y,w,b):
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities)
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
                to fit the data points in x and y
    """

    #Number of training examples
    m = x.shape[0]

    totalCost = 0

    # Variable to keep track of sum of cost from each example
    costSum = 0

    for i in range(m):
        f_wb_i = x[i]*w + b
        cost = (f_wb_i - y[i])**2

        # Add to sum of cost for each example
        costSum += cost

    totalCost = (1 / (2*m))*costSum

    return totalCost
def computeGradient(x,y,w,b):
    """
    Computes the gradient for linear regression
    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities)
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model
    Returns
        dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
        dj_db (scalar): The gradient of the cost w.r.t. the parameter b
    """

    # Number of training examples
    m = x.shape[0]

    # You need to return the following variables correctly
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        # Prediction f_wb for the ith example
        f_wb = x[i]*w + b

        # The gradient for w from the ith example
        dj_dw_i = (f_wb - y[i]) * x[i]

        # The gradient for b from the ith example
        dj_db_i = f_wb - y[i]

        # Uptade the variables
        dj_db += dj_db_i
        dj_dw += dj_dw_i

    # Divide both dj_dw and dj_db by m
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw,dj_db
def gradientDescent(x,y,wIn,bIn,costFunction,gradientFunction,alpha,numIters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iters gradient steps with learning rate alpha

    Args:
      x :    (ndarray): Shape (m,)
      y :    (ndarray): Shape (m,)
      w_in, b_in : (scalar) Initial values of parameters of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """

    # number of training examples
    m = len(x)

    # An array to store cost J and w's at each iteration — primarily for graphing later
    J_history = []
    w_history = []
    w = copy.deepcopy(wIn)  # avoid modifying global w within function
    b = bIn

    for i in range(numIters):

        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradientFunction(x, y, w, b)

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            cost = costFunction(x, y, w, b)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(numIters / 10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")

    # Return w and J,w history for graphing
    return w, b, J_history, w_history

if __name__ == '__main__':
    whichCityToOpenANewFranchise()