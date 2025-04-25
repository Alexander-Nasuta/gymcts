from typing import Any

import matplotlib.pyplot as plt
import numpy as np

CEND = "\33[0m"
CBOLD = "\33[1m"
CITALIC = "\33[3m"
CURL = "\33[4m"
CBLINK = "\33[5m"
CBLINK2 = "\33[6m"
CSELECTED = "\33[7m"

CBLACK = "\33[30m"
CRED = "\33[31m"
CGREEN = "\33[32m"
CYELLOW = "\33[33m"
CBLUE = "\33[34m"
CCYAN = '\33[96m'
CMAGENTA = '\033[35m'
CVIOLET = "\33[35m"
CBEIGE = "\33[36m"
CWHITE = "\33[37m"

CBLACKBG = "\33[40m"
CREDBG = "\33[41m"
CGREENBG = "\33[42m"
CYELLOWBG = "\33[43m"
CBLUEBG = "\33[44m"
CVIOLETBG = "\33[45m"
CBEIGEBG = "\33[46m"
CWHITEBG = "\33[47m"

CGREY = "\33[90m"
CRED2 = "\33[91m"
CGREEN2 = "\33[92m"
CYELLOW2 = "\33[93m"
CBLUE2 = "\33[94m"
CCYAN2 = "\033[36m"
CVIOLET2 = "\33[95m"
CBEIGE2 = "\33[96m"
CWHITE2 = "\33[97m"

CGREYBG = "\33[100m"
CREDBG2 = "\33[101m"
CGREENBG2 = "\33[102m"
CYELLOWBG2 = "\33[103m"
CBLUEBG2 = "\33[104m"
CVIOLETBG2 = "\33[105m"
CBEIGEBG2 = "\33[106m"
CWHITEBG2 = "\33[107m"


def rgb_color_sequence(r: int | float, g: int | float, b: int | float,
                       *, format_type: str = 'foreground') -> str:
    """
    generates a color-codes, that change the color of text in console outputs.

    rgb values must be numbers between 0 and 255 or 0.0 and 1.0.

    :param r:               red value.
    :param g:               green value
    :param b:               blue value

    :param format_type:     specifies weather the foreground-color or the background-color shall be adjusted.
                            valid options: 'foreground','background'
    :return:                a string that contains the color-codes.
    """
    # type: ignore # noqa: F401
    if format_type == 'foreground':
        f = '\033[38;2;{};{};{}m'.format  # font rgb format
    elif format_type == 'background':
        f = '\033[48;2;{};{};{}m'.format  # font background rgb format
    else:
        raise ValueError(f"format {format_type} is not defined. Use 'foreground' or 'background'.")
    rgb = [r, g, b]

    if isinstance(r, int) and isinstance(g, int) and isinstance(b, int):
        if min(rgb) < 0 and max(rgb) > 255:
            raise ValueError("rgb values must be numbers between 0 and 255 or 0.0 and 1.0")
        return f(r, g, b)
    if isinstance(r, float) and isinstance(g, float) and isinstance(b, float):
        if min(rgb) < 0 and max(rgb) > 1:
            raise ValueError("rgb values must be numbers between 0 and 255 or 0.0 and 1.0")
        return f(*[int(n * 255) for n in [r, g, b]])


def wrap_with_color_codes(s: object, /, r: int | float, g: int | float, b: int | float, **kwargs) \
        -> str:
    """
    stringify an object and wrap it with console color codes. It adds the color control sequence in front and one
    at the end that resolves the color again.

    rgb values must be numbers between 0 and 255 or 0.0 and 1.0.

    :param s: the object to stringify and wrap
    :param r: red value.
    :param g: green value.
    :param b: blue value.
    :param kwargs: additional argument for the 'DisjunctiveGraphJspVisualizer.rgb_color_sequence'-method.
    :return:
    """
    return f"{rgb_color_sequence(r, g, b, **kwargs)}" \
           f"{s}" \
           f"{CEND}"


def wrap_evenly_spaced_color(s: Any, n_of_item: int, n_classes: int, c_map="rainbow") -> str:
    """
    Wraps a string with a color scale (a matplotlib c_map) based on the n_of_item and n_classes.
    This function is used to color code the available actions in the MCTS tree visualisation.
    The children of the MCTS tree are colored based on their action for a clearer visualisation.

    :param s: the string (or object) to be wrapped. objects are converted to string (using the __str__ function).
    :param n_of_item: the index of the item to be colored. In a mcts tree, this is the (parent-)action of the node.
    :param n_classes: the number of classes (or items) to be colored. In a mcts tree, this is the number of available actions.
    :param c_map: the colormap to be used (default is 'rainbow').
                  The colormap can be any matplotlib colormap, e.g. 'viridis', 'plasma', 'inferno', 'magma', 'cividis'.
    :return: a string that contains the color-codes (prefix and suffix) and the string s in between.
    """
    if s is None or n_of_item is None or n_classes is None:
        return s

    c_map = plt.cm.get_cmap(c_map)  # select the desired cmap
    arr = np.linspace(0, 1, n_classes + 1)  # create a list with numbers from 0 to 1 with n items

    color_vals = c_map(arr[n_of_item])[:-1]
    color_asni = rgb_color_sequence(*color_vals, format_type='foreground')

    return f"{color_asni}{s}{CEND}"


def wrap_with_color_scale(s: str, value: float, min_val: float, max_val: float, c_map=None) -> str:
    """
    Wraps a string with a color scale (a matplotlib c_map) based on the value, min_val, and max_val.

    :param s: the string to be wrapped
    :param value: the value to be mapped to a color
    :param min_val: the minimum value of the scale
    :param max_val: the maximum value of the scale
    :param c_map: the colormap to be used (default is 'rainbow')
    :return:
    """
    if s is None or min_val is None or max_val is None or min_val >= max_val:
        return s

    if c_map is not None:
        c_map = plt.cm.get_cmap(c_map)  # select the desired cmap
    else:
        from matplotlib.colors import LinearSegmentedColormap
        colors = [
            np.array([255 / 255, 100 / 255, 128 / 255, 1.0]),  # RGBA values
            np.array([63 / 255, 197 / 255, 161 / 255, 1.0]),  # RGBA values
        ]
        c_map = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

    color_vals = c_map((value - min_val) / (max_val - min_val))[:-1]
    color_asni = rgb_color_sequence(*color_vals, format_type='foreground')

    return f"{color_asni}{s}{CEND}"


if __name__ == '__main__':
    res = wrap_with_color_scale("test", 1.0, 0, 1)
    print(res)
