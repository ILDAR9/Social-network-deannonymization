import os
import pylab as plt
import pandas as pd
import networkx as nx
import random
import numpy as np
import pickle
import itertools as it
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, rcParams
from matplotlib.ticker import FormatStrFormatter

base_folder = '/home/ildar/projects/pycharm/social_network_revealing/graphmatching/'
folder_data = os.path.join(base_folder, 'data')
folder_gen = os.path.join(folder_data, 'generated')

FLG_SAVE_IMG = True
pref_g1 = 'g1_'
pref_g2 = 'g2_'
OUT_FOLDER_NAME = '.'


def set_up_plot(d):
    axes = plt.axes()
    axes.autoscale_view()
    axes.set_title(d['label'], fontsize=d['fontsize'])
    plt.legend(loc='upper right', fancybox=True, shadow=True)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    leg = plt.legend()
    leg_lines = leg.get_lines()
    leg_texts = leg.get_texts()
    plt.setp(leg_lines, linewidth=2)
    plt.setp(leg_texts, fontsize=d['fontsize'])

    axes.tick_params(axis='x', labelsize=20)
    axes.tick_params(axis='y', labelsize=20)


def plot_of(d):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 15
    fig_size[1] = 11
    plt.rcParams["figure.figsize"] = fig_size

    plt.xlabel(d['xlabel'], fontsize=d['fontsize'])
    plt.ylabel(d['ylabel'], fontsize=d['fontsize'])
    for prefix in (pref_g1, pref_g2):
        plt.plot(d[prefix + 'x'], d[prefix + 'y'], alpha=d[prefix + 'alpha'], markersize=10,
                 marker=d[prefix + 'marker'], label=d[prefix + 'label'], color=d[prefix + 'color'])
    set_up_plot(d)
    plt.grid()
    plt.ylim(ymax=d['ymax'])
    plt.xlim(xmax=d['xmax'])
    if FLG_SAVE_IMG: plt.savefig(os.path.join(OUT_FOLDER_NAME, d['img_name']))
    plt.show()


def plot_degs(g1, g2=None):
    get_x_y = lambda t: zip(*[(left, count) for left, _, count in t.bins()])
    xs_inst, ys_inst = get_x_y(g1.degree_distribution(bin_width=5))
    xs_vk, ys_vk = get_x_y(g1.degree_distribution(bin_width=1))

    d = {
        'ylabel': 'Distribution',
        'img_name': 'deg_dist.pdf',
        'ymax': None,
        'xmax': 300,
        'xlabel': 'Friends counts',
        'label': 'Degree Distribution',
        'fontsize': 30,
        pref_g1 + 'color': 'r',
        pref_g2 + 'color': 'b',
        pref_g1 + 'marker': 'o',
        pref_g2 + 'marker': 'x',
        pref_g1 + 'alpha': 1,
        pref_g2 + 'alpha': 1,
        pref_g1 + 'label': 'INSTAGRAM',
        pref_g2 + 'label': 'VK',
        pref_g1 + 'x': xs_inst,
        pref_g2 + 'x': xs_vk,
        pref_g1 + 'y': ys_inst,
        pref_g2 + 'y': ys_vk
    }

    plot_of(d)


def plote3d(df, name='precision'):
    rcParams.update({'font.size': 32})
    folder_save = '/home/ildar/projects/pycharm/social_network_revealing/graphmatching/scripts_plot/out'
    fig = plt.figure(figsize=(15, 12))
    ax = Axes3D(fig)

    surf = ax.plot_trisurf(df.threshold, df.seeds, df[name], cmap=cm.jet, linewidth=0.2, )
    #     cb = fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel('threshold', )
    ax.set_ylabel('seeds')
    #     ax.set_zlabel(name)
    ax.xaxis.labelpad = 30
    ax.yaxis.labelpad = 22
    ax.zaxis.labelpad = 32
    ax.set_aspect(1.4)
    ax.tick_params(axis='z', which='major', pad=15)

    plt.savefig(os.path.join(folder_save, 'phase1_%s.pdf' % name))
    plt.show()