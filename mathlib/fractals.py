"""
Classes for fractals

@author: siddhartha.banerjee

"""

import numpy as np
from math import log, log2
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
        max_iter: int = _MAX_ITER,
    ) -> None:
        """Constructor of the class."""
        assert (n >= 2), 'Value of n has to be greater or equal 2'
        self.__n = n
        self.func = lambda z, c: (z ** self.n + c)
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
        max_iter: int = _MAX_ITER,
    ) -> None:
        """Mandelbrot subclass."""
        super().__init__(
            max_iter=max_iter,
        )

    def image(
        self,
        width: int = _WIDTH,
        height: int = _HEIGHT,
    ) -> Image:
        """Draw fractal image."""
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
                color = 255 - int(m * 255 / self._max_iter)
                # Plot the point
                draw.point([x, y], (color, color, color))

        return im

