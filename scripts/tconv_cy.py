# Reproduce convergence table for Carreau-Yasuda

import numpy as np
from utils import *

tests = [
    (["1"], ["0.2", "0.33", "0.66", "0.9", "1.1", "2"]),
    (["0.25", "0.5", "1", "1.5", "2", "4"], ["2.5"]),
    (["3"], ["0.2", "0.33", "0.66", "1.1", "3", "4"]),
    (["0.25", "0.5", "1", "1.5", "2", "4"], ["4"]),
]
nnn = 6


def n_a_to_l(n, a):
    lok = min((a + 2) / 4, (n + 1) / 4)
    if a <= 2 and n <= 3 and n >= 1 / 3:
        l = lok
    elif a > 2 and n > 3:
        l = 1
    else:
        l = min(n, lok)
    return l


def lambda_tau(n, a, dt):
    l = n_a_to_l(n, a)
    return -np.log(1 + l * dt) / dt


def make_tab():
    tabc = "{l" + "c" * nnn + "}"
    tab = ""
    tab += r"\begin{table}[hbt]" + "\n"
    tab += r"\centering" + "\n"
    tab += r"\begin{tabular}" + tabc + "\n"
    for t in tests:
        tab += r"\toprule" + "\n"
        aa, nn = t
        if len(aa) == 1:
            rows = aa
            cols = nn
            cs = "n"
            rs = "a"
            rv = aa[0]
        else:
            rows = nn
            cols = aa
            cs = "a"
            rs = "n"
            rv = nn[0]
        for c in cols:
            tab += f" & {cs}={float(c):.2f}"
        tab += "\\\\\n"
        tab += r"\midrule" + "\n"
        tab += r"\multirow{ 2}{*}{" + f"{rs}={float(rv):.2f}" + "}"
        slope_d_pred = []
        for c in cols:
            if len(aa) == 1:
                n = c
                a = aa[0]
            else:
                n = nn[0]
                a = c
            fdata = {}
            for solver, si in [("be", 1), ("pseudo", 0)]:
                psicol = 7
                dt = 1
                filename = f"./scripts/data/tconv/carreau_yasuda_n{n}_a{a}_dt{dt}_sp0_full{si}.pkl"
                data = pickle_load(filename)["edata"]
                tt = np.concatenate([[0], np.cumsum(data[:-1, 0])])
                data = np.vstack(
                    (tt, data[:, 1:4].transpose(), data[:, psicol].transpose())
                )
                fdata[solver] = data
            beslope_d, beslope_d_t, beslope_d_d = expfit(
                fdata["be"][0], fdata["be"][-2]
            )  # -1 uses psi metric, -2 uses discrete metric
            psslope_d, psslope_d_t, psslope_d_d = expfit(
                fdata["pseudo"][0], fdata["pseudo"][-2]
            )  # -1 uses psi metric, -2 uses discrete metric
            tab += f" & {-beslope_d:0.2f}, {-psslope_d:0.2f}"
            slope_d_pred.append(-lambda_tau(float(n), float(a), float(dt)))
        tab += "\\\\\n"
        for sd in slope_d_pred:
            tab += f" & {sd:0.2f}"
        tab += "\\\\\n"
    tab += r"\end{tabular}" + "\n"
    tab += (
        r"\caption"
        + r"{Experimental convergence rates for the BE (top left value) and Pseudo (top right value) compared against the theoretical $\lambda_\tau$ (bottom) for the Carreau--Yasuda"
        + r" model for different values of model parameters $n$ and $a$."
        + "}\n"
    )
    tab += r"\label" + "{tab:lambdaconv_carreau_yasuda}\n"
    tab += r"\end{table}" + "\n"
    print(tab)


def expfit(t, d):
    s = 10 * t.shape[0] // 20
    e = 11 * t.shape[0] // 20
    pf = np.polyfit(t[s:e], np.log(d[s:e]), 1)
    s = 9 * t.shape[0] // 20
    e = 12 * t.shape[0] // 20
    return pf[0], t[s:e], np.exp(pf[0] * t[s:e] + pf[1])


make_tab()
