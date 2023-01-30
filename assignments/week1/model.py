import numpy as np


class LinearRegression:
    """
    A linear regression model that uses matrix closed form to fit the model.
    """

    w: np.ndarray
    b: float

    # b means bias? I think, so not sure if we will have bias (the constant intercept) for sure.
    # In this case, I use a parameter that defaults to True.

    def __init__(self, b=True):

        """
        Initialization on necessary parameters in class.

        Arguments:
            w: parameters corresponding to X.
            b: constant bias, default is True.

        Returns:
            Nothing.

        """
        self.w = None
        self.b = b

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:

        """
        Closed form method to calculate weights.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The input result.

        Returns:
            Nothing.

        """
        if self.b:
            row = X.shape[0]
            X = np.hstack((np.ones((row, 1)), X))
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X: np.ndarray) -> np.ndarray:

        """
        Predict the results with new X.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """

        if self.b:
            row = X.shape[0]
            X = np.hstack((np.ones((row, 1)), X))

        y_pred = X @ self.w.T
        return y_pred


class GradientDescentLinearRegression(LinearRegression):

    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def __init__(self, b=True):

        """
        Initialization on necessary parameters in class.

        Arguments:
            w: parameters corresponding to X.
            b: constant bias, default is True.

        Returns:
            Nothing.

        """
        self.w = None
        self.b = b

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:

        """
        Gradient descent method to calculate optimal weights.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The input result.
            lr: learning rate.
            epochs: number of iterations.

        Returns:
            Nothing.

        """

        N, D = X.shape
        if self.b:
            X = np.hstack((np.ones((N, 1)), X))
            self.w = np.zeros((D + 1,))
        else:
            self.w = np.zeros((D,))

        for i in range(epochs):
            gradient = 2 / N * (X.T @ (X @ (self.w) - y))
            self.w -= lr * gradient

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """

        N = X.shape[0]
        X = np.hstack((np.ones((N, 1)), X))

        y_pred = X @ self.w.T
        return y_pred
