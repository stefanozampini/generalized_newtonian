import matplotlib.pyplot as plt

# from brokenaxes import brokenaxes
import numpy as np
from utils import *

plt.rcParams.update({"font.size": 24})
plt.rcParams.update({"lines.markersize": 8})
plt.rcParams.update({"lines.linewidth": 2})
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["text.usetex"] = True

atol = 1.0e-8


def load_data(filename):
    data = pickle_load(filename)["edata"]
    ddata = pickle_load(filename)["ddata"]
    pseudo = np.vstack((data[:, 0:3].transpose(), data[:, -2:].transpose()))
    newton = np.vstack((ddata[:, 0:2].transpose(), ddata[:, 1].transpose()))
    pdata = pickle_load(filename)["pdata"]
    E_inf = pdata["E_exact"]
    Ec_inf = pdata["Ec_exact"]
    if Ec_inf == np.inf and pseudo[-1][-1] < atol:
        Ec_inf = pseudo[2][-1]
    if E_inf == np.inf and pseudo[-1][-1] < atol:
        E_inf = pseudo[1][-1]
    if E_inf == np.inf and newton[-1][-1] < atol:
        E_inf = newton[1][-1]
    return (pseudo, newton, E_inf, Ec_inf)


def make_tab(logt, rows, rowst, cols, colst, model, rowm=None, colm=None, dtinfo=None):
    def _defm(c):
        return c

    if rowm is None:
        rowm = _defm
    if colm is None:
        colm = _defm
    # tabc = "{" + "c" + "l" * len(cols) + "}"
    tabc = "{" + "c" * (len(cols) + 1) + "}"
    tab = ""
    tab += r"\begin{table}[hbt]" + "\n"
    tab += r"\centering" + "\n"
    tab += r"\begin{tabular}" + tabc + "\n"
    tab += r"\toprule" + "\n"
    for c in cols:
        tab += " & " + colst.format(key=colm(c))
    tab += "\\\\\n"
    if dtinfo:
        tab += r"\midrule" + "\n"
        tab += r"$\tau_0$"
        for dt in dtinfo:
            if int(dt) == dt:
                tab += f" & {int(dt)}"
            else:
                tab += f" & {dt:g}"
        tab += "\\\\\n"

    tab += r"\midrule" + "\n"
    names = []
    for r in rows:
        rnames = []
        tab += rowst.format(key=rowm(r))
        for c in cols:
            log = logt.format(row=r, col=c)
            rnames.append(log)
            pseudo, newton, E_inf, Ec_inf = load_data(log)
            pitr = pseudo[-1].shape[0]
            nitr = newton[-1].shape[0]
            if pseudo[-1][-1] > atol:
                pitr = "-"
            if newton[-1][-1] > atol:
                nitr = "-"
            tab += f" & {pitr} ({nitr})"
        tab += "\\\\\n"
        names.append(rnames)
    tab += r"\bottomrule" + "\n"
    tab += r"\end{tabular}" + "\n"
    tab += (
        r"\caption"
        + "{Iteration counts for the Pseudo method compared against Newton (in parentheses) for the "
        + model
        + " model for different model parameters (columns) and refinement levels (rows)."
    )
    if dtinfo:
        tab += r" The initial time step $\tau_0$ is also reported."
    tab += (
        r" The symbol ``-'' indicates failure to converge in 200 iterations with absolute tolerance 1.E-8 for $\|\mathcal{R}^\ast[u_h]\|_{\ell^2}$.}"
        + "\n"
    )
    modell = model.replace(" ", "_").replace("-", "_").lower()
    label = f"comp_newton_{modell}"
    tab += r"\label" + "{tab:" + label + "}\n"
    tab += r"\end{table}" + "\n"
    return tab, names


# powerlaw tabular
rows = [0, 1, 2]
cols = [1.5, 3, 4, 8, 10, 20, 40, 80, 100]
dtinfo = [10, 9, 4, 0.5, 2.0e-1, 1.0e-1, 5.0e-2, 2.0e-2, 1.5e-2]
rowst = "r = {key}"
colst = "p = {key}"
logt = "./scripts/data/compare_newton/powerlaw_p{col}_r{row}.pkl"
tab, pnames = make_tab(logt, rows, rowst, cols, colst, "power-law", dtinfo=dtinfo)
print(tab)

# optimal design
rows = [0, 1, 2]
cols = [0, 1]
rowst = "r = {key}"
colst = r"$\lambda_d$" + " = {key}"
logt = "./scripts/data/compare_newton/optimal_l{col}_r{row}.pkl"
colm = lambda x: 0.0084 if x == 0 else 0.0145
tab, onames = make_tab(logt, rows, rowst, cols, colst, "optimal design", colm=colm)
print(tab)

# Bingham
rows = [0, 1, 2]
cols = [2, 3, 4]
logt = "./scripts/data/compare_newton/bingham_e{col}_r{row}.pkl"
rowst = "r = {key}"
colst = r"$b_\epsilon$ = " + "1.e-{key}"
tab, bnames = make_tab(logt, rows, rowst, cols, colst, "Bingham")
print(tab)


def make_figure(log, savelabel=None):
    pseudo, newton, E_inf, Ec_inf = load_data(log)
    fig, axs = plt.subplots()
    axt = axs.twinx()
    axs.set_xlabel("iteration number")
    pseudo_plot = pseudo[-1][-1] < atol or len(pseudo[-1]) > 1
    newton_plot = newton[-1][-1] < atol or len(newton[-1]) > 1
    if newton_plot:
        axs.plot(
            newton[-1], color="b", marker="d", label=r"$\|\mathcal{R}^*[u^N_{h,n}]\|_2$"
        )
    if pseudo_plot:
        axs.plot(
            pseudo[-2],
            color="r",
            marker="x",
            label=r"$\|\mathcal{R}^{f,*}[c_{h,n},u_{h,n}]\|_2$",
        )
        axs.plot(
            pseudo[-1], color="r", marker="o", label=r"$\|\mathcal{R}^*[u_{h,n}]\|_2$"
        )
    axs.set_yscale("log")
    if pseudo_plot:
        axt.plot(pseudo[0], color="g", marker="v", label=r"$\tau_n$")
    axt.set_yscale("log")
    ncol = 4
    fig.legend(
        fontsize=16,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=ncol,
        frameon=False,
    )
    fig.subplots_adjust(bottom=0.12)
    if savelabel:
        fig.savefig(f"scripts/figures/ncomp_residual_{savelabel}.png")

    fig, axs = plt.subplots()
    axs.set_xlabel("iteration number")
    axs.set_yscale("log")
    if newton_plot:
        axs.plot(
            np.clip(np.abs(newton[0] - E_inf), a_min=1.0e-16, a_max=None),
            color="b",
            marker="d",
            label=r"$E(u^N_{h,n}) - E_\infty$",
        )
    if pseudo_plot:
        axs.plot(
            np.clip(np.abs(pseudo[1] - E_inf), a_min=1.0e-16, a_max=None),
            color="r",
            marker="o",
            label=r"$E(u_{h,n}) - E_\infty$",
        )
        axs.plot(
            np.clip(np.abs(pseudo[2] - Ec_inf), a_min=1.0e-16, a_max=None),
            color="r",
            marker="x",
            label=r"$|\mathcal{E}(c_{h,n}) - \mathcal{E}_\infty|$",
        )
    ncol = 3
    fig.legend(
        fontsize=16,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=ncol,
        frameon=False,
        columnspacing=1.0,
    )
    fig.subplots_adjust(bottom=0.12)
    if savelabel:
        fig.savefig(f"scripts/figures/ncomp_energy_{savelabel}.png")


make_figure("scripts/data/compare_newton/powerlaw_p1.5_r2.pkl", savelabel="p1_5")
make_figure("scripts/data/compare_newton/powerlaw_p100_r2.pkl", savelabel="p100")
