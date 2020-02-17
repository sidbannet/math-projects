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
        self.number_of_terms = 1
        self.func = None
        self.fft = {
            'power': [],
            'freq': [],
        }

    def _next_value_(
        self,
        last_value: float,
        number_of_terms: int,
        function: tuple = None,
    ) -> float:
        """Solve for the next generation, using default function."""
        number_of_terms = self.number_of_terms
        assert (number_of_terms >= 0), "Number of terms can't be negetive."
        sum_of_terms = 0.0
        if function is not None:
            term_func = function
        else:
            term_func = []
            for i in number_of_terms:
                term_func.append(_terms_(n=i))
        for iterms, ifunc in enumerate(term_func):
            sum_of_terms += ifunc(last_value)
        return float(self.growth_rate * sum_of_terms)

    def plot_function(
        self,
        fig_info: tuple = None,
    ) -> tuple:
        """Show the logistic map equation in plots."""
        if fig_info is not None:
            fig, ax = fig_info
        else:
            fig = plt.figure('Fig: Function used')
            ax = fig.subplots()
        x = np.linspace(start=0.0, stop=1.0, num=100)
        y = []
        if self.func is not None:
            func = self.func
        else:
            func = _terms_(n=1)
        for ix, xterm in enumerate(x):
            yterm = 0.0
            try:
                for iterms, ifunc in enumerate(func):
                    yterm += self.next_value(
                        last_value=xterm,
                        growth_rate=self.growth_rate,
                        function=ifunc,
                    )
            except TypeError:
                yterm += self.growth_rate * func(xterm)
            y.append(yterm)
        y = np.asarray(y)
        ax.plot(x, y, linewidth=3, linestyle='-', color='k', label='Function')
        ax.plot(x, x, linewidth=1, linestyle='--', color='k', label='Equality')
        ax.grid(True)
        ax.set(xlim=[0, 1])
        ax.set(ylim=[0, 1])
        ax.set(xlabel='Value of $x_{i}$')
        ax.set(ylabel='Value of $x_{i+1}$')
        ax.legend()
        ax.set(title='Function of the logistic map equation')
        return fig, ax


    def solve(
        self,
        number_of_generations: int = 100,
        number_of_terms: int = 1,
    ):
        """Solve for generations and march for values in logistic map.
        :param number_of_terms: number of terms in the series
        :param number_of_generations: number of generations done in solve
        """
        assert (number_of_generations > 1), "Number of generation is too low."
        self.func = []
        for i in range(number_of_terms + 1):
            self.func.append(_terms_(n=i))
        for i in range(number_of_generations):
            self.x_values.append(
                self._next_value_(
                    last_value=self.x_values[-1],
                    number_of_terms=number_of_terms,
                    function=self.func,
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
            fig = plt.figure('Fig: Value marching with generations')
            ax = fig.subplots(1, 2)
        if fig_prop is None:
            fig_prop = {
                'color': 'k',
                'linewidth': 2,
                'linestyle': '-',
                'marker': '.',
                'fillstyle': 'full',
                'xlabel': 'Generations',
                'ylabel': 'Value',
                'freqlabel': 'Time Period',
                'amplitudelabel': 'Log of Power',
                'title': 'Progression of the logistic map equation',
                'freqtitle': 'Power frequency spectrum',
                'grid': True,
                'suptitle': 'Logistic Map',
            }

        ax[0].plot(
            self.x_values,
            color=fig_prop['color'],
            linewidth=fig_prop['linewidth'],
            linestyle=fig_prop['linestyle'],
            marker=fig_prop['marker'],
            fillstyle=fig_prop['fillstyle'],
        )
        ax[0].set(xlabel=fig_prop['xlabel'])
        ax[0].set(ylabel=fig_prop['ylabel'])
        ax[0].set(title=fig_prop['title'])
        ax[0].grid(fig_prop['grid'])

        ax[1].plot(
            self.fft['freq'][1:],
            np.log(self.fft['power'][1:]),
            color=fig_prop['color'],
            linewidth=fig_prop['linewidth'],
            linestyle=fig_prop['linestyle'],
            marker='None',
        )
        ax[1].set(xlabel=fig_prop['freqlabel'])
        ax[1].set(ylabel=fig_prop['amplitudelabel'])
        ax[1].set(title=fig_prop['freqtitle'])
        ax[1].grid(fig_prop['grid'])
        ax[1].set(xlim=[0, 1])

        fig.suptitle(fig_prop['suptitle'])

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
