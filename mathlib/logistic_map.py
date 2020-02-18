"""
Classes for logistic map

@author: siddhartha.banerjee

"""

import sympy
import numpy as np
import matplotlib.pyplot as plt

_DEFAULT_GROWTH_RATE = float(3.4)
_DEFAULT_INITIAL_VALUE = float(0.5)


class Marching:
    """Marching with generations."""

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
        self.y = None
        self.func = None
        self.fft = {
            'power': [],
            'freq': [],
        }

    def _next_value_(
            self,
            last_value: float,
            number_of_terms: int,
            function: list = None,
    ) -> float:
        """Solve for the next generation, using default function."""

        number_of_terms = self.number_of_terms
        assert (number_of_terms >= 0), "Number of terms can't be negative."
        sum_of_terms = 0.0
        if function is not None:
            term_func = function
        else:
            term_func = []
            for i in range(number_of_terms + 1):
                term_func.append(_terms_(n=i))
        for index, individual_term_function in enumerate(term_func):
            sum_of_terms += individual_term_function(last_value)
        return float(self.growth_rate * sum_of_terms)

    def plot_function(
            self,
            fig_info: tuple = None,
    ) -> tuple:
        """
        Show the logistic map equation in plots.
        Returns tuple with fig, ax

        """

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
        self.y = np.asarray(y)
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
        """
        Solve for generations and march for values in logistic map.

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
        """
        Plot the progression of values against generations.
        Returns tuple with fig, ax

        """

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


class Map(Marching):
    """Marching with generations with a given mapping function."""

    def __init__(
            self,
            initial_value: float = _DEFAULT_INITIAL_VALUE,
            growth_rate: float = _DEFAULT_GROWTH_RATE,
            function_for_mapping: list = None,
            number_of_terms_for_default_parabola: int = 1,
    ):
        """Instantiate the subclass with given function."""

        super().__init__(
            initial_value=initial_value,
            growth_rate=growth_rate,
        )
        self.func = function_for_mapping
        self.number_of_terms = number_of_terms_for_default_parabola


class Bifurcation:
    """Class to generate bifurcation."""

    def __init__(
        self,
        function_to_map: list = None,
    ):
        """Instantiate the class."""

        self.y_equilibrium = np.empty([])
        self.r_equilibrium = np.empty([])
        self.function_to_map = function_to_map

    def __call__(self, *args, **kwargs):
        """The object is called like function."""

        number_of_generations_for_equilibrium = 0
        if args is not None:
            for argv in args:
                number_of_generations_for_equilibrium += int(argv)
        else:
            number_of_generations_for_equilibrium = 150
        self._get_equilibrium_values(
            number_of_generations=number_of_generations_for_equilibrium
        )

    def _get_equilibrium_values(
        self,
        number_of_generations: int = 300,
    ) -> None:
        """Get the values of equilibrium"""

        if number_of_generations <= 90:
            raise ValueError('Number of generations \
             to equilibrium is too low.')

        n_generations = number_of_generations
        n_equilibrium = 64

        demo_map_obj = Map(
            growth_rate=1.0,
            function_for_mapping=self.function_to_map,
        )
        _, _ = demo_map_obj.plot_function()
        growth_rate = np.linspace(
            start=0.1, stop=float(1.0 / demo_map_obj.y.max()), num=1000
        )
        for index, irate in enumerate(growth_rate):
            map_obj = Map(
                function_for_mapping=self.function_to_map,
                initial_value=0.1,
                growth_rate=irate,
            )
            map_obj.solve(number_of_generations=n_generations)
            y_values = np.unique(
                np.around(map_obj.x_values[-n_equilibrium:-1], decimals=3)
            )
            self.y_equilibrium = np.append(
                self.y_equilibrium, y_values
            )
            self.r_equilibrium = np.append(
                self.r_equilibrium,
                np.full_like(y_values, irate, dtype=np.double)
            )


def _terms_(
        n: int,
):
    """Terms in the logistic map function."""

    return lambda x: (
                             (x ** n) * (1 - (x ** n))
                     ) / sympy.factorial(n)
