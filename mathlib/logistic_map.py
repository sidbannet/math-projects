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
            number_of_terms: int = 1,
            map_function: list = None,
    ):
        """Initialize the class."""

        assert (initial_value > 0), "Initial value is equal or less than 0."
        assert (initial_value < 1), "Initial value is equal or greater than 1."
        assert (growth_rate > 0), "Growth rate is less than or equal to 0."
        self.growth_rate = growth_rate
        self.x_values = [initial_value]
        self.number_of_terms = number_of_terms
        self.y = None
        self.func = map_function
        self.fft = {
            'power': [],
            'freq': [],
        }

    def __call__(self, *args, **kwargs) -> tuple:
        """Call the class."""

        try:
            number_of_generations, fig, ax = args
            fig_prop = kwargs
        except ValueError:
            number_of_generations = 100
            fig = None
            ax = None
            fig_prop = {}
        self.solve(number_of_generations=number_of_generations)
        fig, ax, fig_prop = self.plots(fig, ax, fig_prop)
        return fig, ax, fig_prop

    def _next_value_(
            self,
            last_value: float,
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
            *fig_info,
    ) -> tuple:
        """
        Show the logistic map equation in plots.
        Returns tuple with fig, ax

        """

        try:
            fig, ax = fig_info
        except ValueError:
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
            number_of_terms: int = None,
    ) -> None:
        """
        Solve for generations and march for values in logistic map.

        :param number_of_terms: number of terms in the series
        :param number_of_generations: number of generations done in solve

        """

        if number_of_terms is not None:
            self.number_of_terms = number_of_terms
        assert (number_of_generations > 1), "Number of generation is too low."
        if self.func is None:
            self.func = []
            for i in range(self.number_of_terms + 1):
                self.func.append(_terms_(n=i))
        for i in range(number_of_generations):
            self.x_values.append(
                self._next_value_(
                    last_value=self.x_values[-1],
                    function=self.func,
                )
            )
        self.fft['power'] = np.abs(np.fft.fft(self.x_values, axis=0))
        self.fft['freq'] = list(
            np.array(range(len(self.x_values))) / len(self.x_values)
        )

    def plots(
            self,
            *fig_info,
            **fig_prop,
    ) -> tuple:
        """
        Plot the progression of values against generations.
        Returns tuple with fig, ax

        """

        try:
            fig, ax = fig_info
        except ValueError:
            fig = plt.figure('Fig: Value marching with generations')
            ax = fig.subplots(1, 2)
        if 'color' not in fig_prop.keys():
            fig_prop['color'] = 'k'
        if 'linewidth' not in fig_prop.keys():
            fig_prop['linewidth'] = 2
        if 'linestyle' not in fig_prop.keys():
            fig_prop['linestyle'] = '-'
        if 'marker' not in fig_prop.keys():
            fig_prop['marker'] = '.'
        if 'fillstyle' not in fig_prop.keys():
            fig_prop['fillstyle'] = 'full'
        if 'xlabel' not in fig_prop.keys():
            fig_prop['xlabel'] = 'Generations'
        if 'ylabel' not in fig_prop.keys():
            fig_prop['ylabel'] = 'Value'
        if 'freqlabel' not in fig_prop.keys():
            fig_prop['freqlabel'] = 'Time Period'
        if 'amplitudelabel' not in fig_prop.keys():
            fig_prop['amplitudelabel'] = 'Log of Power'
        if 'title' not in fig_prop.keys():
            fig_prop['title'] = 'Progression of logistic map equation'
        if 'freqtitle' not in fig_prop.keys():
            fig_prop['freqtitle'] = 'Power frequency spectrum'
        if 'suptitle' not in fig_prop.keys():
            fig_prop['suptitle'] = 'Logistic Map'
        if 'grid' not in fig_prop.keys():
            fig_prop['grid'] = True

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


class LogisticMap(Marching):
    """Marching with generations for logistic map function."""

    def __init__(
            self,
            initial_value: float = _DEFAULT_INITIAL_VALUE,
            growth_rate: float = _DEFAULT_GROWTH_RATE,
    ):
        """Instantiate the class."""

        super().__init__(
            initial_value=initial_value,
            growth_rate=growth_rate,
            map_function=_terms_(n=1),
            number_of_terms=1,
        )


class Bifurcation:
    """Class to generate bifurcation."""

    def __init__(
            self,
            map_obj: Marching = None,
    ):
        """Instantiate the class."""

        self.y_equilibrium = np.empty([])
        self.r_equilibrium = np.empty([])
        try:
            self.map_obj = map_obj
            self.function_to_map = map_obj.func
            self.num_of_terms = map_obj.number_of_terms
        except AttributeError:
            self.map_obj = LogisticMap(initial_value=0.1, growth_rate=1.0)
            self.function_to_map = self.map_obj.func
            self.num_of_terms = self.map_obj.number_of_terms

    def __call__(self, *args, **kwargs):
        """The object is called like function."""

        number_of_generations_for_equilibrium = 0
        if args is not None:
            for argv in args:
                number_of_generations_for_equilibrium += int(argv)
        else:
            number_of_generations_for_equilibrium = 150
        try:
            self.get_equilibrium_values(
                number_of_generations=number_of_generations_for_equilibrium,
            )
        except ValueError:
            self.get_equilibrium_values(
                number_of_generations=150
            )

        self.plots()

    def get_equilibrium_values(
            self,
            number_of_generations: int = 300,
    ) -> None:
        """Get the values of equilibrium"""

        if number_of_generations <= 90:
            raise ValueError('Number of generations \
             to equilibrium is too low.')

        n_generations = number_of_generations
        n_equilibrium = 64

        demo_map_obj = self.map_obj
        _, ax = demo_map_obj.plot_function()
        ax.set(title='Base function of the logistic map equation')
        growth_rate = np.linspace(
            start=0.1, stop=float(1.0 / demo_map_obj.y.max()), num=1000
        )
        for index, irate in enumerate(growth_rate):
            map_obj = Marching(
                initial_value=0.1,
                growth_rate=irate,
                map_function=demo_map_obj.func,
                number_of_terms=demo_map_obj.number_of_terms,
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

    def plots(
            self,
            *fig_info,
            **fig_prop,
    ) -> tuple:
        """Plot the bifurcation diagram."""

        try:
            fig, ax = fig_info
        except ValueError:
            fig = plt.figure('Fig: Bifurcation diagram')
            ax = fig.subplots()
        if 'color' not in fig_prop.keys():
            fig_prop['color'] = 'k'
        if 'marker' not in fig_prop.keys():
            fig_prop['marker'] = '.'
        if 'fillstyle' not in fig_prop.keys():
            fig_prop['fillstyle'] = 'full'
        if 'xlabel' not in fig_prop.keys():
            fig_prop['xlabel'] = 'Growth Rate'
        if 'ylabel' not in fig_prop.keys():
            fig_prop['ylabel'] = 'Equilibrium value'
        if 'title' not in fig_prop.keys():
            fig_prop['title'] = 'Progression of logistic map equation'
        if 'grid' not in fig_prop.keys():
            fig_prop['grid'] = True
        ax.plot(
            self.r_equilibrium,
            self.y_equilibrium,
            '.',
            color=fig_prop['color'],
        )
        ax.grid(fig_prop['grid'])
        ax.set(xlabel=fig_prop['xlabel'])
        ax.set(ylabel=fig_prop['ylabel'])
        ax.set_ylim(0, 1)
        ax.set(title=fig_prop['title'])

        return fig, ax, fig_prop


def _terms_(
        n: int,
):
    """Terms in the logistic map function."""

    return lambda x: (
                             (x ** n) * (1 - (x ** n))
                     ) / sympy.factorial(n)
