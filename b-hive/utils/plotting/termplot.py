import numpy as np
import termplotlib as tpl
from sklearn.metrics import roc_curve, auc
from scipy.special import softmax

import logging
import matplotlib.pyplot as plt

# Set the logging level to WARNING to suppress DEBUG and INFO messages
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def terminal_roc(
    predictions, truth, title=None, truth_index=0, veto_index=None, xlabel=None
):
    # check if sum of logits == 1.
    if np.abs(np.mean(np.sum(predictions, axis=-1)) - 1) > 1e-3:
        predictions = softmax(predictions, axis=-1)
#    if len(predictions.shape) == 1:
#        bvsl = predictions
#    else:
#        b_pred = predictions[:, :1].sum(axis=-1)
#        l_pred = predictions[:, 1:].sum(axis=-1)
#        bvsl = np.where((b_pred + l_pred) > 0, (b_pred) / (b_pred + l_pred), -1)

    ttbb_pred = predictions[:,0]
    ttbj_pred = predictions[:,1]
    ttcc_pred = predictions[:,2]
    ttcj_pred = predictions[:,3]
    ttother_pred = predictions[:,4]

    ttbbvsall = np.where( (ttbb_pred+ttbj_pred+ttcc_pred+ttcj_pred+ttother_pred) > 0, (ttbb_pred) / (ttbb_pred+ttbj_pred+ttcc_pred+ttcj_pred+ttother_pred), -1)
    ttccvsall = np.where( (ttbb_pred+ttbj_pred+ttcc_pred+ttcj_pred+ttother_pred) > 0, (ttcc_pred) / (ttbb_pred+ttbj_pred+ttcc_pred+ttcj_pred+ttother_pred), -1)
#    if np.sum(np.unique(truth)) > 1:
#        b_jets = truth == 0
#        if not (veto_index is None):
#            veto = truth != veto_index
#        veto = np.ones(len(b_jets), dtype=bool)
#    else:
#        b_jets = truth
#        if not (veto_index is None):
#            veto = np.ones(truth.shape, dtype=bool)
#        else:
#            veto = np.ones(len(b_jets), dtype=bool)

    if "ttbb" in xlabel:
        if np.sum(np.unique(truth)) > 1:
            ttbb_ev = truth == 0
            if not (veto_index is None):
                veto = truth != veto_index
            veto = np.ones(len(ttbb_ev), dtype=bool)
        else:
            ttbb_ev = truth
            if not (veto_index is None):
                veto = np.ones(truth.shape, dtype=bool)
            else:
                veto = np.ones(len(ttbb_ev), dtype=bool)
        try:
            fig = tpl.figure()
            label = ["ttbb vs all"]
            fpr, tpr, _ = roc_curve(ttbb_ev[veto], ttbbvsall[veto])
            fig.plot(
                tpr,
                fpr,
                width=90,
                height=30,
                xlim=(0.1, 1),
                ylim=(0.0001, 1),
                label=label,
                xlabel=xlabel,
                title=title,
                extra_gnuplot_arguments=["set ylabel mis-id", "set logscale y"],
            )
            fig.show()
        except FileNotFoundError as e:
            print(e)
            print("Is gnuplot installed on your machine?")

    if "ttcc" in xlabel:
        if np.sum(np.unique(truth)) > 1:
            ttcc_ev = truth == 0
            if not (veto_index is None):
                veto = truth != veto_index
            veto = np.ones(len(ttcc_ev), dtype=bool)
        else:
            ttcc_ev = truth
            if not (veto_index is None):
                veto = np.ones(truth.shape, dtype=bool)
            else:
                veto = np.ones(len(ttcc_ev), dtype=bool)
        try:
            fig = tpl.figure()
            label = ["ttcc vs all"]
            fpr, tpr, _ = roc_curve(ttcc_ev[veto], ttccvsall[veto])
            fig.plot(
                tpr,
                fpr,
                width=90,
                height=30,
                xlim=(0.1, 1),
                ylim=(0.0001, 1),
                label=label,
                xlabel=xlabel,
                title=title,
                extra_gnuplot_arguments=["set ylabel mis-id", "set logscale y"],
            )
            fig.show()
        except FileNotFoundError as e:
            print(e)
            print("Is gnuplot installed on your machine?")


def _term_roc(disc, truth, label, xlabel, title):
    fig = tpl.figure()
    fpr, tpr, _ = roc_curve(truth, disc)
    fig.plot(
        tpr,
        fpr,
        width=90,
        height=30,
        xlim=(0.1, 1),
        ylim=(0.0001, 1),
        label=label,
        xlabel=xlabel,
        title=title,
        extra_gnuplot_arguments=["set ylabel mis-id", "set logscale y"],
    )
    fig.show()
