"""
Classes for fractals

@author: siddhartha.banerjee

"""

import numpy as np
from math import log, log2
from PIL import Image, ImageDraw
from IPython.display import Image as ImageDisp


class Multibrot:
    """Multibrot fractals."""

    _MAX_ITER = 80  # Potential function cutoff
    # Image size (pixels)
    _WIDTH = 1200
    _HEIGHT = 800

    def __init__(
        self,
        n: int = 2,
        _height: int = _HEIGHT,
        _width: int = _WIDTH,
        _max_iter: int = _MAX_ITER,
    ) -> None:
        """Constructor of the class."""
        assert (n >= 2), 'Value of n has to be greater or equal 2'
        # Plot window
        _RE_START = -2
        _RE_END = 1
        _IM_START = -1
        _IM_END = 1

        self.__n = n
        self.func = lambda z, c: (z ** self.n + c)
        self._height = _height
        self._width = _width
        self._Re_window = [_RE_START, _RE_END]
        self._Im_window = [_IM_START, _IM_END]
        self._max_iter = _max_iter

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
