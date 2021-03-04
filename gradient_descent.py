import numpy as np


class SigmoidNeuronExample:
    def __init__(self, X, Y, **kwargs):
        self.X = X
        self.Y = Y
        self.weight = kwargs.get("initial_weight", -2)
        self.bias = kwargs.get("initial_bias", -2)
        self.eta = kwargs.get("eta", 1.0)
        self.max_epochs = kwargs.get("max_epochs", 1000)
        self.do_gradient_descent()

    def f(self, x):
        return 1.0 / (1.0 + np.exp(-(self.weight * x + self.bias)))

    def error(self):
        err = 0.0
        for x, y in zip(self.X, self.Y):
            fx = self.f(x)
            err += (fx - y) ** 2
        err = err / len(self.X)
        return err

    def grad_b(self, x, y):
        fx = self.f(x)
        return (fx - y) * fx * (1 - fx)

    def grad_w(self, x, y):
        fx = self.f(x)
        return (fx - y) * fx * (1 - fx) * x

    def do_gradient_descent(self):
        for i in range(self.max_epochs):
            dw, db = 0, 0
            for x, y in zip(X, Y):
                dw += self.grad_w(x, y)
                db += self.grad_b(x, y)
            self.weight -= self.eta * dw
            self.bias -= self.eta * db


if __name__ == "__main__":
    X = [0.5, 2.5]
    Y = [0.2, 0.9]
    example = SigmoidNeuronExample(
        X=X, Y=Y, initial_weight=-2, initial_bias=-2, eta=1.0, max_epochs=1000
    )
