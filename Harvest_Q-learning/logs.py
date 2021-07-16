import json
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns

from source.env.lib.enums import Neon


def plot(data, inds=None, label='data', c=Neon.RED.norm, lw=3, linestyle='-', dashes=None):
    if inds is None:
        inds = np.arange(len(data))
    sns.despine(offset=10, trim=True)
    sns.set_style("darkgrid", {'xtick.bottom': True,
                               'ytick.left': True,
                               })
    ax = sns.lineplot(x=inds, y=data, color=c, linewidth=lw, label=label, linestyle=linestyle, dashes=dashes, ci=95)
    ax.lines[-1].set_linestyle(linestyle)
    ax.legend(handlelength=3)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 3e6)
    if dashes is not None:
        ax.lines[-1].set_dashes(dashes)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: str(int(x // 1e6)) +
                                                                                    ('' if x == 0 else 'M') if x % 1e6 == 0 else
                                       str(x / 1e6) + 'M'))
    ax.legend(loc='lower left', ncol=1, handlelength=2, prop={'size': 24})
    #ax.legend(loc='lower left', ncol=2, handlelength=1, prop={'size': 20})
    #ax.legend(loc='lower left', ncol=2, handlelength=1, prop={'size': 24})
   # ax.legend(loc='right', ncol=1, bbox_to_anchor=(1.5, 1.5), handlelength=1, prop={'size': 32})


def dark():
    plt.style.use('dark_background')


def labels(xlabel='x', ylabel='y', title='title',
           axsz=24, titlesz=28):
    pass
   # plt.xlabel(xlabel, fontsize=axsz)


#  plt.ylabel(ylabel, fontsize=axsz)
#  plt.title(title, fontsize=titlesz)


def axes(ac, tc):
    ax = plt.gca()
    ax.title.set_color(ac)
    ax.xaxis.label.set_color(ac)
    ax.yaxis.label.set_color(ac)
    for spine in ax.spines.values():
        spine.set_color(tc)


def limits(xlims=None, ylims=None):
    if xlims is not None:
        plt.xlim(*xlims)

    if ylims is not None:
        plt.ylim(*ylims)


def ticks(ts, tc):
    ax = plt.gca()
    ax.tick_params(axis='x', colors=tc)
    ax.tick_params(axis='y', colors=tc)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(ts)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(ts)
        tick.label1.set_fontweight('bold')


def legend(ts, tc):
    leg = plt.legend(loc='upper right')
    for text in leg.get_texts():
        plt.setp(text, color=tc)
        plt.setp(text, fontsize=ts)


def fig():
    fig = plt.gcf()
    fig.set_size_inches(12, 12, forward=True)
    # plt.tight_layout()


def show():
    fig.canvas.set_window_title('Projekt Godsword')


def save(fPath):
    fig = plt.gcf()
    fig.savefig(fPath, dpi=200, bbox_inches='tight', pad_inches=0)


def load(fDir):
    try:
        with open(fDir, 'r') as f:
            logs = json.load(f)

        logDict = defaultdict(list)
        for log in logs:
            for k, v in log.items():
                logDict[k].append(v)
        return logDict
    except Exception as e:
        print(e)
        return None


def godsword():
    labels('Steps', 'Value', 'Projekt: Godsword')
    axes(Neon.BLACK.norm, Neon.BLACK.norm)
    ticks(24, Neon.BLACK.norm)
    # legend(18, Neon.CYAN.norm)
    fig()


def plots(logs):
    colors = Neon.color12()
    logs = reversed([e for e in logs.items()])
    for idx, kv in enumerate(logs):
        k, v = kv
        color = colors[idx].norm
        plot(v, k, color)


def log():
    fDir = 'resource/logs/'
    fName = 'frag.png'
    logs = load(fDir + 'logs.json')
    dark()
    plots(logs)
    plt.ylim(0, 150)
    godsword()
    save(fDir + fName)
    plt.close()
