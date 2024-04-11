import numpy as np

class DualNumber:
    def __init__(self, value, derivative):
        self.value = value
        self.derivative = derivative

    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.value + other.value, self.derivative + other.derivative)
        else:
            return DualNumber(self.value + other, self.derivative)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.value - other.value, self.derivative - other.derivative)
        else:
            return DualNumber(self.value - other, self.derivative)

    def __rsub__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(other.value - self.value, other.derivative - self.derivative)
        else:
            return DualNumber(other - self.value, -self.derivative)

    def __mul__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.value * other.value, self.value * other.derivative + self.derivative * other.value)
        else:
            return DualNumber(self.value * other, self.derivative * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.value / other.value, (self.derivative * other.value - self.value * other.derivative) / (other.value ** 2))
        else:
            return DualNumber(self.value / other, self.derivative / other)

    def __rtruediv__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(other.value / self.value, (other.derivative * self.value - other.value * self.derivative) / (self.value ** 2))
        else:
            return DualNumber(other / self.value, -other * self.derivative / (self.value ** 2))

    def __pow__(self, power):
        if isinstance(power, DualNumber):
            raise ValueError("Exponentiation with DualNumber as power is not supported.")
        else:
            return DualNumber(self.value ** power, power * self.value ** (power - 1) * self.derivative)

    def __repr__(self):
        return f"DualNumber(value={self.value}, derivative={self.derivative})"

def sin(x):
    if isinstance(x, DualNumber):
        if isinstance(x.value, DualNumber):
            return DualNumber(np.sin(x.value.value), np.cos(x.value.value) * x.value.derivative * x.derivative)
        else:
            return DualNumber(np.sin(x.value), np.cos(x.value) * x.derivative)
    else:
        return np.sin(x)

def cos(x):
    if isinstance(x, DualNumber):
        if isinstance(x.value, DualNumber):
            return DualNumber(np.cos(x.value.value), -np.sin(x.value.value) * x.value.derivative * x.derivative)
        else:
            return DualNumber(np.cos(x.value), -np.sin(x.value) * x.derivative)
    else:
        return np.cos(x)

def exp(x):
    if isinstance(x, DualNumber):
        if isinstance(x.value, DualNumber):
            return DualNumber(np.exp(x.value.value), np.exp(x.value.value) * x.value.derivative * x.derivative)
        else:
            return DualNumber(np.exp(x.value), np.exp(x.value) * x.derivative)
    else:
        return np.exp(x)

def log(x):
    if isinstance(x, DualNumber):
        if isinstance(x.value, DualNumber):
            return DualNumber(np.log(x.value.value), x.value.derivative * x.derivative / x.value.value)
        else:
            return DualNumber(np.log(x.value), x.derivative / x.value)
    else:
        return np.log(x)

def sqrt(x):
    if isinstance(x, DualNumber):
        if isinstance(x.value, DualNumber):
            return DualNumber(np.sqrt(x.value.value), 0.5 * x.value.derivative * x.derivative / np.sqrt(x.value.value))
        else:
            return DualNumber(np.sqrt(x.value), 0.5 * x.derivative / np.sqrt(x.value))
    else:
        return np.sqrt(x)

def tanh(x):
    if isinstance(x, DualNumber):
        if isinstance(x.value, DualNumber):
            tanh_value = np.tanh(x.value.value)
            return DualNumber(tanh_value, (1 - tanh_value ** 2) * x.value.derivative * x.derivative)
        else:
            tanh_value = np.tanh(x.value)
            return DualNumber(tanh_value, (1 - tanh_value ** 2) * x.derivative)
    else:
        return np.tanh(x)

def f(x):
    print(f"f: x = {x}")
    result = sin(x) + x ** 2 * cos(x) + exp(x) * log(x) + sqrt(x) * tanh(x)
    print(f"f: result = {result}")
    return result

def grad_f(x):
    print(f"grad_f: x = {x}")
    x_dual = DualNumber(x, 1.0)
    result = f(x_dual)
    print(f"grad_f: f(x_dual) = {result}")
    return result.derivative

def hessian_f(x):
    print(f"hessian_f: x = {x}")
    x_dual = DualNumber(x, 1.0)
    grad_dual = grad_f(x_dual)
    print(f"hessian_f: grad_f(x_dual) = {grad_dual}")
    hessian_value = grad_dual.derivative
    print(f"hessian_f: hessian_value = {hessian_value}")
    return hessian_value

def jacobian(func, x):
    x_dual = [DualNumber(xi, np.eye(len(x))[i]) for i, xi in enumerate(x)]
    return np.array([func(x_dual)[i].derivative for i in range(len(x))])

def main():
    x = 2.0
    
    # Compute the function value, gradient, and Hessian using automatic differentiation
    function_value = f(x)
    gradient_value = grad_f(x)
    hessian_value = hessian_f(x)
    
    print(f"Function value at x = {x}: {function_value}")
    print(f"Gradient value at x = {x}: {gradient_value}")
    print(f"Hessian value at x = {x}: {hessian_value}")
    
    # Compute the Jacobian of a vector-valued function
    def vector_func(x):
        component1 = DualNumber(x[0].value ** 2 + sin(x[1]).value,
                                x[0].derivative * 2 * x[0].value + cos(x[1]).value * x[1].derivative)
        component2 = DualNumber(exp(x[0]).value * cos(x[1]).value,
                                exp(x[0]).value * cos(x[1]).derivative * x[1].derivative + exp(x[0]).derivative * cos(x[1]).value * x[0].derivative)
        return np.array([component1, component2])
    
    x_vector = np.array([1.0, 2.0])
    jacobian_value = jacobian(vector_func, x_vector)
    
    print(f"Jacobian of the vector-valued function at x = {x_vector}:")
    print(jacobian_value)

if __name__ == "__main__":
    main()