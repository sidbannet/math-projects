"""
Classes for logistic map

@author: siddhartha.banerjee

"""

import sympy
import numpy as np


class Marching:
    """Marching with iterations."""

    _DEFAULT_GROWTH_RATE = float(3.4)
    _DEFAULT_INITIAL_VALUE = float(0.5)

    def __init__(
            self,
            growth_rate: float = _DEFAULT_GROWTH_RATE,
            initial_value: float = _DEFAULT_INITIAL_VALUE,
    ):
        """Initialize the class."""
        assert (initial_value > 0), "Initial value is equal or less than 0."
        assert (initial_value < 1), "Initial value is equal or greater than 1."
        assert (growth_rate > 0), "Growth rate is less than or equal to 0."
        self.growth_rate = growth_rate
        self.x_values = [initial_value]

    def _next_value_(
            self,
            last_value: float,
            number_of_terms: int = 1,
    ) -> float:
        """Solve for the next iteration."""
        assert (number_of_terms >= 0), "Number of terms can't be negetive."
        sum_of_terms = 0.0
        term_func = []
        for i in range(number_of_terms + 1):
            term_func.append(_terms_(n=i))
            sum_of_terms += term_func[i](last_value)
        return float(self.growth_rate * sum_of_terms)

    @staticmethod
    def next_value(
        last_value: float = 0.5,
        growth_rate: float = 1.0,
        function=np.sin,
    ) -> float:
        """Return value based on passed function."""
        return growth_rate * function(last_value)

    def solve(
        self,
        number_of_iterations: int = 100,
    ):
        """Solve for iterations and march for values in logistic map.
        :param number_of_iterations: number of iterations done in solve
        """
        assert (number_of_iterations > 1), "Number of iteration is too low."
        for i in range(number_of_iterations):
            self.x_values.append(
                self._next_value_(
                    last_value=self.x_values[-1],
                    number_of_terms=1,
                )
            )


def _terms_(
        n: int,
) -> float:
    """Terms in the logistic map function."""

    return lambda x: (
        (x ** n) * (1 - (x ** n))
    ) / sympy.factorial(n)
