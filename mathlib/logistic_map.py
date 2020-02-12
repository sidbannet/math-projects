"""
Classes for logistic map

@author: siddhartha.banerjee

"""

import sympy


class Marching:
    """Marching with iterations."""

    _DEFAULT_INITIAL_VALUE = float(0.5)

    def __init__(
            self,
            growth_rate: float = 3.4,
            initial_value: float = _DEFAULT_INITIAL_VALUE,
    ):
        """Initialize the class."""
        self.growth_rate = growth_rate
        self.x_values = [initial_value]

    def _next_value_(
            self,
            last_value: float,
            number_of_terms: int = 1,
    ):
        """Solve for the next iteration."""
        sum_of_terms = 0.0
        term_func = []
        for i in range(number_of_terms + 1):
            term_func.append(_terms_(n=i))
            sum_of_terms += term_func[i](last_value)
        return float(self.growth_rate * sum_of_terms)


def _terms_(
        n: int,
) -> float:
    """Terms in the logistic map function."""

    return lambda x: (
        (x ** n) * (1 - (x ** n))
    ) / sympy.factorial(n)
