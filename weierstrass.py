import os
import time
from typing import List, Tuple, Dict
import math

import matplotlib.pyplot as plt
from pylab import mpl


def weierstrass(domain: Tuple[float, float] = (0.0, 10.0), division_round: float = 2, 
                max_n: int = 100, a: float = 0.5, b: int = 13) -> Dict[float, float]:
    '''
    the Weierstrass function

    @param domain: the domain of the function, default: (0.0, 10.0), a closed interval
    @param division_round: the round digit of the division value, default: 2, i.e. the division value is 10 ** (-2)
    @param max_n: the maximum n of the function, as an approximate infinity for the sum operation, default: 100
    @param a: the a parameter of the function, 0 < a < 1, default: 0.5
    @param b: the b parameter of the function, default: 13

    @return values: the value dictionary, key: x, value: y
    '''
    
    time_start = time.time()

    # check: if the a value and b value is conform to the mathematical condition
    if not a * b > 1 + (3 / 2) * math.pi:
        raise Exception(f'the value of a * b should be larger than 1 + (3 / 2) * PI')

    values = {}
    x = domain[0]
    while x <= domain[1]:
        y = 0
        for n in range(max_n + 1):
            y += math.pow(a, n) * math.cos(math.pow(b, n) * math.pi * x)
        values[round(x, division_round)] = y

        x += 10 ** (- division_round)

    time_end = time.time()
    time_calculate = round(time_end - time_start, 3)
    print(f'Weierstrass function calculation time: {time_calculate} s', '\n')

    return values


def cosine(domain: Tuple[float, float] = (0.0, 2 * math.pi), division_round: float = 2, 
           amplitude: float = 1, omega: float = 1) -> Dict[float, float]:
    '''
    the cosine function

    @param domain: the domain of the function, default: (0.0, 2 * math.pi), a closed interval
    @param division_round: the round digit of the division value, default: 2, i.e. the division value is 10 ** (-2)
    @param amplitude: the amplitude parameter of the function, default: 1
    @param omega: the omega parameter of the function, i.e. y = amplitude * cos(omega * x), default: 1

    @return values: the value dictionary, key: x, value: y
    '''

    time_start = time.time()

    values = {}
    x = domain[0]
    while x <= domain[1]:
        values[round(x, division_round)] = amplitude * math.cos(omega * x)
        x += 10 ** (- division_round)

    time_end = time.time()
    time_calculate = round(time_end - time_start, 3)
    print(f'cosine function calculation time: {time_calculate} s', '\n')

    return values


def draw_curves(x_values: List[float], y_values: Dict[str, List[float]], 
                x_interval: int = 1, y_margin_ratios: Tuple[float, float] = (0.1, 0.3), 
                file_path: str = 'curve.png'):
    '''
    draw curves by matplotlib, and save the picture

    @param x_values: the x values
    @param y_values: the y values, key: function names, value: function values
    @param x_interval: the label interval of the x axis
    @param y_margin_ratios: the margin ratios of the y axis, based on the range of y values
    @param file_path: the file path of the picture
    '''

    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False  # display Chinese characters normally

    # set resolution ratio
    plt.figure(dpi=300)

    # draw the curves
    colours = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'grey']
    i = 0
    for function_name, function_values in y_values.items():
        plt.plot(x_values, function_values, label=function_name, linewidth=0.5, color=colours[i % len(colours)])
        i += 1

    # only keep the interval scales of the x-axis scales
    x_values_axis, x_scales = [], []
    for x in x_values:
        if x == int(x) and not x % x_interval:
            x_values_axis.append(x)
            x_scales.append(round(x))
    plt.xticks(ticks=x_values_axis, labels=x_scales)

    # the axis labels
    plt.xlabel(xlabel='x', fontsize=12)
    plt.ylabel(ylabel='y', fontsize=12)

    # the lower and upper limit of the y-axis data
    y_min = min(min(y_values[key]) for key in y_values)
    y_max = max(max(y_values[key]) for key in y_values)
    y_range = y_max - y_min
    y_lb = y_min - y_range * y_margin_ratios[0]
    y_ub = y_max + y_range * y_margin_ratios[1]
    plt.ylim(y_lb, y_ub)

    # the location and shape of the legend
    plt.legend(loc='best', ncol=3)

    # save the picture
    plt.savefig(file_path)


if __name__ == '__main__':
    domain = (0.0, 4.0)
    division_round = 2
    print()

    # Weierstrass function values
    weierstrass_values = weierstrass(domain=domain, division_round=division_round, max_n=275)

    # cosine function values
    cosine_values = cosine(domain=domain, division_round=division_round, amplitude=2, omega=math.pi)

    # get x values and check y values
    x_values = set()
    x_values_wei = set(weierstrass_values.keys())
    x_values_cos = set(cosine_values.keys())
    for x in x_values_wei:
        if x in x_values_cos:
            x_values.add(x)
    x_values = sorted(x_values)
    y_values = {
        'Weierstrass': [weierstrass_values[x] for x in x_values], 
        'cosine': [cosine_values[x] for x in x_values]
    }

    picture_path = os.path.join(os.path.dirname(__file__), 'weierstrass.png')
    draw_curves(x_values=x_values, y_values=y_values, file_path=picture_path)
