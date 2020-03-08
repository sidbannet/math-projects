"""
Classes for fractals

@author: siddhartha.banerjee

"""

import numpy as np
from math import log, log2
import sympy as sym
from PIL import Image, ImageDraw

# GLOBAL CONSTANTS
_MAX_ITER = 80  # Potential function cutoff
# Image size (pixels)
_WIDTH = 1200
_HEIGHT = 800
# Plot window
_RE_START = -2
_RE_END = 1
_IM_START = -1
_IM_END = 1


class Multibrot:
    """Multibrot fractals."""

    def __init__(
        self,
        n: int = 2,
        alpha: float = 1.0,
        max_iter: int = _MAX_ITER,
    ) -> None:
        """Constructor of the class."""
        assert (n >= 1), 'Value of n has to be greater or equal 1'
        self.__n = n
        self.__alpha = alpha
        self.func = lambda z, c: (z ** self.n + self.alpha * c)
        self._Re_window = [_RE_START, _RE_END]
        self._Im_window = [_IM_START, _IM_END]
        self._max_iter = max_iter

    @property
    def n(self) -> int:
        """Getter of property n."""
        return self.__n

    @n.setter
    def n(self, n) -> None:
        """Setter of property n."""
        try:
            assert (n >= 2), 'Value of n has to be greater or equal 2'
            assert (type(n) is int), 'n is integer'
            self.__n = n
        except TypeError as te:
            print(te)
            raise Exception('Give integer value greater than 1')

    @property
    def alpha(self) -> float:
        """Getter of property n."""
        return self.__alpha

    @alpha.setter
    def alpha(self, alpha) -> None:
        """Setter of property alpha."""
        try:
            assert (alpha > 0.0), 'Value of alpha has to be greater or equal 0'
            assert (type(alpha) is float), 'alpha is float'
            self.__alpha = alpha
        except TypeError as te:
            print(te)
            raise Exception('Give floating value greater than 0')

    def make_newton(
        self,
        p: sym.core = None,
    ) -> None:
        """Make this a Newton Fractal."""
        try:
            f = sym.utilities.lambdify(
                sym.Symbol('z'), p / sym.diff(p)
            )
        except NameError as ne:
            print(ne)
            raise Exception('Symbol name should be \'z\'')
        except ZeroDivisionError as ze:
            print(ze)
            raise Exception('Cannot divide by zero')
        self.func = lambda z, c: (z ** self.n + self.alpha * (f(z) + c))

    def potential(
        self,
        c: complex,
        z0: complex = (0 + 0j),
    ) -> tuple:
        """Gives potential function"""
        z = z0
        n_p = 0
        while np.abs(z) < np.abs(self._Re_window[0]) and n_p < self._max_iter:
            z = self.func(z=z, c=c)
            n_p += 1
        if n_p == self._max_iter:
            return n_p, n_p
        else:
            return n_p, n_p + 1 - log(log2(abs(z)))


class Mandelbrot(Multibrot):
    """Mandelbrot fractal class."""

    def __init__(
        self,
        n: int = 2,
        alpha: float = 1.0,
        max_iter: int = _MAX_ITER,
    ) -> None:
        """Mandelbrot subclass."""
        super().__init__(
            n=n,
            alpha=alpha,
            max_iter=max_iter,
        )

    def image(
        self,
        width: int = _WIDTH,
        height: int = _HEIGHT,
        color: bool = False,
    ) -> Image:
        """Draw fractal image."""
        if color:
            im = Image.new('HSV', (width, height), (0, 0, 0))
        elif not color:
            im = Image.new('RGB', (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(im)

        for x in range(0, width):
            for y in range(0, height):
                # Convert pixel coordinate to complex number
                c = complex(
                    self._Re_window[0]
                    + (x / width) * (self._Re_window[1] - self._Re_window[0]),
                    self._Im_window[0]
                    + (y / height) * (self._Im_window[1] - self._Im_window[0])
                )
                # Compute the number of iterations
                m, _ = self.potential(c=c)
                # The color depends on the number of iterations
                # And plot the point
                if color:
                    hue = int(255 * m / self._max_iter)
                    saturation = 255
                    value = 255 if m < self._max_iter else 0
                    draw.point([x, y], (hue, saturation, value))
                elif not color:
                    value = 255 - int(m * 255 / self._max_iter)
                    draw.point([x, y], (value, value, value))

        if color:
            im = im.convert('RGB')
        return im


class JuliaSet(Multibrot):
    """Julia set."""

    def __init__(
        self,
        c: complex = None,
        n: int = 2,
        alpha: float = 1.0,
        max_iter: int = _MAX_ITER,
    ):
        """Instantiate the subclass Julia Set."""
        assert (c is not None), "Give a complex value to instance."
        super().__init__(
            n=n,
            alpha=alpha,
            max_iter=max_iter,
        )
        try:
            self.__c__ = c
        except ValueError as ve:
            print(ve)
            raise Exception('Complex value cannot be assigned.')

    @property
    def c(
        self,
    ) -> complex:
        """Getter for the complex value."""
        return self.__c__

    @c.setter
    def c(self, c: complex = None) -> None:
        """Setter for the complex value."""
        assert (type(c) is complex), 'Enter complex number.'
        try:
            self.__c__ = c
        except ValueError as ve:
            print(ve)
            raise Exception('Cannot set the complex value.')

    def image(
        self,
        width: int = _WIDTH,
        height: int = _HEIGHT,
        color: bool = False,
    ) -> Image:
        """Draw fractal image."""
        if color:
            im = Image.new('HSV', (width, height), (0, 0, 0))
        elif not color:
            im = Image.new('RGB', (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(im)

        for x in range(0, width):
            for y in range(0, height):
                # Convert pixel coordinate to complex number
                z0 = complex(
                    self._Re_window[0]
                    + (x / width) * (self._Re_window[1] - self._Re_window[0]),
                    self._Im_window[0]
                    + (y / height) * (self._Im_window[1] - self._Im_window[0])
                )
                # Compute the number of iterations
                m, _ = self.potential(c=self.c, z0=z0)
                # The color depends on the number of iterations
                # And plot the point
                if color:
                    hue = int(255 * m / self._max_iter)
                    saturation = 255
                    value = 255 if m < self._max_iter else 0
                    draw.point([x, y], (hue, saturation, value))
                elif not color:
                    value = 255 - int(m * 255 / self._max_iter)
                    draw.point([x, y], (value, value, value))

        if color:
            im = im.convert('RGB')
        return im


class NewtonFractal(JuliaSet):
    """Newton fractals."""

    def __init__(
        self,
        n: int = 1,
        alpha: float = - 1.0,
        p: sym.core = None,
        max_iter: int = _MAX_ITER,
    ):
        """Instantiate NewtonFractals."""
        super().__init__(n=n, alpha=alpha, max_iter=max_iter, c=0)
        self.__p = p
        self.make_newton(p=p)

    def roots(
        self,
    ) -> list:
        """Return solution root."""
        out = roots(self.__p)
        return out


def linear_interpolation(value1, value2, t):
    """Linear interpolation."""
    return value1 * (1 - t) + value2 * t


def roots(f) -> list:
    """Return roots."""
    out = []
    try:
        sol = sym.solve(f)
    except SyntaxError as se:
        print(se)
        raise Exception('Syntax error')
    for individual_sol in sol:
        out.append(
            complex(sym.N(individual_sol))
        )
    return out
