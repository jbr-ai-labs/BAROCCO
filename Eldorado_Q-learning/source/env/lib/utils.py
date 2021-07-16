# Bounds checker
def inBounds(r, c, shape, border=0):
    R, C = shape
    return (border < r < R - border and
            border < c < C - border)
