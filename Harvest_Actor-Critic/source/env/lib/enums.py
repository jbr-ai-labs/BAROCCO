# Various Enums used for handling materials, entity types, etc.
# Data texture pairs are used for enums that require textures.
# These textures are filled in by the Render class at run time.

import colorsys
from enum import Enum

import numpy as np


class Defaults:
    BLACK = (0, 0, 0)
    GRAY3 = (20, 20, 20)
    GRAY2 = (40, 40, 40)
    GRAY1 = (60, 60, 60)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    GOLD = (212, 175, 55)
    MASK = (214, 127, 255)


def rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgbNorm(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


class Color:
    def __init__(self, name, hexVal):
        self.name = name
        self.hex = hexVal
        self.rgb = rgb(hexVal)
        self.norm = rgbNorm(hexVal)
        self.value = self.rgb  # Emulate enum

    def packet(self):
        return self.hex


def makeColor(idx, h=1, s=1, v=1):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    rgbval = tuple(int(255 * e) for e in [r, g, b])
    hexval = '%02x%02x%02x' % rgbval
    return Color(str(idx), hexval)


class Color256:
    def make256():
        parh, parv = np.meshgrid(np.linspace(0.075, 1, 16), np.linspace(0.25, 1, 16)[::-1])
        parh, parv = parh.T.ravel(), parv.T.ravel()
        idxs = np.arange(256)
        params = zip(idxs, parh, parv)
        colors = [makeColor(idx, h=h, s=1, v=v) for idx, h, v in params]
        return colors

    colors = make256()


class Neon:
    RED = Color('RED', '#800000')
    ORANGE = Color('ORANGE', '#f58231')
    YELLOW = Color('YELLOW', '#ffff00')

    GREEN = Color('GREEN', '#006400')
    MINT = Color('MINT', '#00ff80')
    CYAN = Color('CYAN', '#4169E1')

    BLUE = Color('BLUE', '#000080')
    PURPLE = Color('PURPLE', '#8000ff')
    MAGENTA = Color('MAGENTA', '#ff00ff')

    FUCHSIA = Color('FUCHSIA', '#ff0080')
    SPRING = Color('SPRING', '#80ff80')
    SKY = Color('SKY', '#0080ff')

    WHITE = Color('WHITE', '#ffffff')
    GRAY = Color('GRAY', '#666666')
    BLACK = Color('BLACK', '#000000')

    BLOOD = Color('BLOOD', '#bb0000')
    BROWN = Color('BROWN', '#8B4513')
    GOLD = Color('GOLD', '#eec600')
    SILVER = Color('SILVER', '#b8b8b8')

    TERM = Color('TERM', '#41ff00')
    MASK = Color('MASK', '#d67fff')

    DARKGREEN = Color('DARKGREEN', '#006400')

    def color12():
        return (
            Neon.RED, Neon.PURPLE, Neon.BROWN, Neon.GREEN,
            Neon.BLUE, Neon.CYAN,
            Neon.GOLD, Neon.PURPLE, Neon.MAGENTA,
            Neon.FUCHSIA, Neon.SPRING, Neon.SKY)

    def rand12():
        twelveColor = color12()
        randInd = np.random.randint(0, len(twelveColor))
        return twelveColor[randInd]


class Palette:
    def __init__(self, n):
        self.n = n
        if n <= 12:
            self.colors = Neon.color12()
        else:
            self.colors = Color256.colors

    def color(self, idx):
        if self.n > 12:
            idx = int(idx * 256 // self.n)
        return self.colors[idx]
