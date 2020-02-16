"""
Classes for logistic map

@author: siddhartha.banerjee

"""

import sympy
import numpy as np
import matplotlib.pyplot as plt


class Marching:
    """Marching with generations."""

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
        self.fft = {
            'power': [],
            'freq': [],
        }

    def _next_value_(
        self,
        last_value: float,
        number_of_terms: int = 1,
    ) -> float:
        """Solve for the next generation."""
        assert (number_of_terms >= 0), "Number of terms can't be negetive."
        sum_of_terms = 0.0
        term_func = []
        for i in range(number_of_terms + 1):
            term_func.append(_terms_(n=i))
            sum_of_terms += term_func[i](last_value)
        return float(self.growth_rate * sum_of_terms)

    def solve(
        self,
        number_of_generations: int = 100,
    ):
        """Solve for generations and march for values in logistic map.
        :param number_of_generations: number of generations done in solve
        """
        assert (number_of_generations > 1), "Number of generation is too low."
        for i in range(number_of_generations):
            self.x_values.append(
                self._next_value_(
                    last_value=self.x_values[-1],
                    number_of_terms=1,
                )
            )
        self.fft['power'] = np.abs(np.fft.fft(self.x_values, axis=0))
        self.fft['freq'] = list(
            np.array(range(len(self.x_values))) / len(self.x_values)
        )

    def plots(
        self,
        fig_info: tuple = None,
        fig_prop: dict = None,
    ) -> tuple:
        """Plot the progression of values against generations."""
        if fig_info is not None:
            fig, ax = fig_info
        else:
            fig = plt.figure('Value marching')
            ax = fig.subplots()
        if fig_prop is None:
            fig_prop = {
                'color': 'k',
                'linewidth': 2,
                'linestyle': '-',
                'marker': '.',
                'fillstyle': 'full',
                'xlabel': 'Generations',
                'ylabel': 'Value',
                'title': 'Progression of the logistic map equation',
                'grid': True,
            }

        ax.plot(
            self.x_values,
            color=fig_prop['color'],
            linewidth=fig_prop['linewidth'],
            linestyle=fig_prop['linestyle'],
            marker=fig_prop['marker'],
            fillstyle=fig_prop['fillstyle'],
        )
        ax.set(xlabel=fig_prop['xlabel'])
        ax.set(ylabel=fig_prop['ylabel'])
        ax.set(title=fig_prop['title'])
        ax.grid(fig_prop['grid'])

        return fig, ax, fig_prop


    @staticmethod
    def next_value(
            last_value: float = 0.5,
            growth_rate: float = 1.0,
            function=np.sin,
    ) -> float:
        """Return value based on passed function."""
        return growth_rate * function(last_value)


def _terms_(
    n: int,
) -> float:
    """Terms in the logistic map function."""

    return lambda x: (
        (x ** n) * (1 - (x ** n))
    ) / sympy.factorial(n)
