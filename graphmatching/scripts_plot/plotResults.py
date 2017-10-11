#!/usr/bin/env python
import pylab as pl
import os

# IN_FOLDER = "_erdos_renyi_s_1"
IN_FOLDER = "gm"

DATA_FOLDER_NAME = os.path.join('data_plot', IN_FOLDER)
OUT_FOLDER_NAME = 'out'
PLOT_TITLE = 'User Identity Linkage (VK, Instagram)'
FLG_SAVE_IMG = True
MAX_X = 200
MAX_Y = 25000

def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for row in f:
            a = tuple(float(num) for num in row.split())
            if a[0] <= MAX_X: data.append(a)
    return data


def set_up():
    def set_up_folders():
        check_and_create_foldeer = lambda dname: os.makedirs(dname) if not os.path.exists(dname) else None
        check_and_create_foldeer(OUT_FOLDER_NAME)
        check_and_create_foldeer(os.path.join(OUT_FOLDER_NAME, DATA_FOLDER_NAME))

    if FLG_SAVE_IMG: set_up_folders()


def plot_data():
    pref_gm1 = 'gm1_'
    pref_gm2 = 'gm2_'
    d = {pref_gm1 + 'label': 'ExpandUserLinkage on topology',
         pref_gm1 + 'color': 'red',
         pref_gm1 + 'marker': 'o',
         pref_gm2 + 'label': 'ExpandUserLinkage on topology and node name',
         pref_gm2 + 'color': 'blue',
         pref_gm2 + 'marker': 'x',
         'xlabel': 'Initial seeds'}

    def set_up_plot():
        axes = pl.axes()
        axes.autoscale_view()
        axes.set_title(PLOT_TITLE)
        pl.legend(loc='upper right', fancybox=True, shadow=True)
        pl.ylim(ymin=0)
        pl.xlim(xmin=0)

    def plot_of():
        pl.xlabel(d['xlabel'])
        pl.ylabel(d['ylabel'])
        for prefix in (pref_gm1, pref_gm2):
            data_file = os.path.join(DATA_FOLDER_NAME, d[prefix + 'fname'])
            pl.plot(*zip(*read_data(data_file)), markersize=10, marker=d[prefix + 'marker'], label=d[prefix + 'label'],
                       color=d[prefix + 'color'])
        set_up_plot()
        pl.grid()
        pl.ylim(ymax=d['ymax'])
        pl.gcf().canvas.set_window_title(d['ylabel'])
        if FLG_SAVE_IMG: pl.savefig(os.path.join(OUT_FOLDER_NAME, DATA_FOLDER_NAME, d['img_name']))
        pl.show()

    def plot_precision():
        d.update({
            pref_gm1 + 'fname': pref_gm1 + 'precision.txt',
            pref_gm2 + 'fname': pref_gm2 + 'precision.txt',
            'ylabel': 'precision',
            'img_name': 'precision.png',
            'ymax': 1})
        plot_of()

    def plot_recall():
        d.update({
            pref_gm1 + 'fname': pref_gm1+'recall.txt',
            pref_gm2 + 'fname': pref_gm2+'recall.txt',
            'ylabel': 'recall',
            'img_name': 'recall.png',
            'ymax': 1})
        plot_of()

    def plot_count():
        d.update({
            pref_gm1 + 'fname': pref_gm1+'correct.txt',
            pref_gm2 + 'fname': pref_gm2+'correct.txt',
            'ylabel': 'count',
            'img_name': 'count.png',
            'ymax': MAX_Y})
        plot_of()

    def plot_matches():
        d.update({
            pref_gm1 + 'fname': pref_gm1+'matched.txt',
            pref_gm2 + 'fname': pref_gm2+'matched.txt',
            'ylabel': 'matches',
            'img_name': 'matches.png',
            'ymax': MAX_Y})
        plot_of()

    def plot_f1_score():
        d.update({
            pref_gm1 + 'fname': pref_gm1+'f1-score.txt',
            pref_gm2 + 'fname': pref_gm2+'f1-score.txt',
            'ylabel': 'F1-score',
            'img_name': 'F1-score.png',
            'ymax': 1})
        plot_of()

    # plot_precision()
    # plot_count()
    # plot_matches()
    plot_f1_score()
    # plot_recall()

######
# MAIN
######
set_up()
plot_data()