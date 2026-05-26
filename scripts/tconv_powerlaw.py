# Generates figures for powerlaw convergence
import matplotlib.pyplot as plt
import numpy as np
from utils import *

plt.rcParams.update({"font.size": 24})
plt.rcParams.update({"lines.linewidth": 2})
# plt.rcParams.update({"lines.markersize": 8})
# plt.rcParams["figure.figsize"] = (12.0, 9.0)
plt.rcParams["text.usetex"] = True

pps = ["1.067", "1.1", "1.2", "1.334", "1.5", "3", "4", "5", "6"]
dts = ["0.25", "0.5", "1"]
vector = False
# vector = True
# save = True
save = False


def p_to_l(p):
    if p < 4 / 3:
        l = p - 1
    elif p > 4:
        l = 1
    else:
        l = p / 4
    return l


def ltp(p, dt):
    if isinstance(p, float):
        l = p_to_l(p)
    else:
        l = np.array([p_to_l(pp) for pp in p])
    return -np.log(1 + l * dt) / dt


def load_data(vector, solver):
    if vector:
        vector = 1
        psicol = 7
    else:
        vector = 0
        psicol = 6
    if solver == "full":
        full = 1
        split = 0
    elif solver == "pseudo":
        full = 0
        split = 0
    elif solver == "split":
        full = 0
        split = 1
    else:
        RuntimeError("solver {solver}")

    out = {}
    for dti, dt in enumerate(dts):
        outt = {}
        for pi, p in enumerate(pps):
            filename = f"./scripts/data/tconv/powerlaw_p{p}_v{vector}_dt{dt}_sp{split}_full{full}.pkl"
            data = pickle_load(filename)["edata"]
            tt = np.concatenate([[0], np.cumsum(data[:-1, 0])])
            data = np.vstack(
                (tt, data[:, 1:3].transpose(), data[:, psicol].transpose())
            )
            outt[p] = data
        out[dt] = outt

    return out


def expfit(t, d, wh=None):
    s = 10 * t.shape[0] // 20
    e = 11 * t.shape[0] // 20
    pf = np.polyfit(t[s:e], np.log(d[s:e]), 1)
    s = 9 * t.shape[0] // 20
    e = 12 * t.shape[0] // 20
    return pf[0], t[s:e], np.exp(pf[0] * t[s:e] + pf[1])


full = load_data(vector, "full")
pseudo = load_data(vector, "pseudo")
split = load_data(vector, "split")

# representative
dtrep = dts[-1]
data = full[dtrep]
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for i, (p, ps) in enumerate(zip(["1.334", "4"], ["4/3", "4"])):
    pdata = data[p]
    axt = axs[i].twinx()

    if float(p) < 2:
        ediff = pdata[2] - pdata[1]
    else:
        ediff = pdata[1] - pdata[2]
    if vector:  # disk is not exactly integrated
        ediff = np.abs(ediff)
    ediff = np.clip(ediff, a_min=1.0e-16, a_max=None)
    slope_e, slope_e_t, slope_e_d = expfit(pdata[0], ediff)
    slope_d, slope_d_t, slope_d_d = expfit(pdata[0], pdata[3])
    slope_d_pred = ltp(float(p), float(dtrep))
    slope_d_mid = len(slope_d_t) // 2
    slope_e_mid = len(slope_e_t) // 2

    axs[i].set_xlabel("$t$")
    axs[i].set_title(f"$p = {ps}$")
    if i == 0:
        axs[i].plot(pdata[0][1:], pdata[1][1:], color="b", label=r"$E(u_{h,n})$")
        axs[i].plot(
            pdata[0][1:], pdata[2][1:], color="r", label=r"$\mathcal{E}(c_{h,n})$"
        )
        axt.plot(
            pdata[0][1:],
            ediff[1:],
            color="m",
            linestyle=":",
            label=r"$|\mathcal{E}(c_{h,n}) - E(u_{h,n})|$",
        )
        axt.plot(
            pdata[0][1:],
            pdata[3][1:],
            color="g",
            linestyle="--",
            label=r"$d(c_{h,n},c^\ast)$",
        )
        axt.plot(slope_d_t, slope_d_d * 10, color="g", linestyle=":")
        axt.text(
            slope_d_t[slope_d_mid],
            100 * slope_d_d[slope_d_mid],
            rf"${-slope_d:0.2} \, (\lambda_\tau = {-slope_d_pred:0.2})$",
            fontsize=18,
        )
    else:
        axs[i].plot(pdata[0][1:], pdata[1][1:], color="b")
        axs[i].plot(pdata[0][1:], pdata[2][1:], color="r")
        axt.plot(pdata[0][1:], ediff[1:], linestyle=":", color="m")
        axt.plot(pdata[0][1:], pdata[3][1:], linestyle="--", color="g")
        axt.plot(slope_d_t, slope_d_d * 10, color="g", linestyle=":")
        axt.text(
            slope_d_t[slope_d_mid],
            100 * slope_d_d[slope_d_mid],
            rf"${-slope_d:0.2} \, (\lambda_\tau = {-slope_d_pred:0.2})$",
            fontsize=18,
        )
    axt.set_yscale("log")
fig.legend(loc="upper center", ncol=4, frameon=False)
fig.tight_layout()
fig.subplots_adjust(top=0.78, bottom=0.14)
if save:
    case = "scalar" if vector is False else "vector"
    fig.savefig(f"scripts/figures/tconv_repr_{case}.png")
fig.show()


def get_slopes(data, p, vector):
    if float(p) < 2:
        ediff = data[2] - data[1]
    else:
        ediff = data[1] - data[2]
    if vector:  # disk is not exactly integrated
        ediff = np.abs(ediff)
    ediff = np.clip(ediff, a_min=1.0e-16, a_max=None)
    slope_e, slope_e_t, slope_e_d = expfit(data[0], ediff)
    slope_d, slope_d_t, slope_d_d = expfit(data[0], data[3])
    return slope_e, slope_d


upper = 1
lambda_t_p = [
    np.linspace(1, 1.33333, 100 // 15),
    np.linspace(4 / 3, 4, 800 // 15),
    np.linspace(4.01, 6, 200 // 5),
]
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
kwargs = {"markersize": 8}
for i, dt in enumerate(dts):
    fdata = full[dt]
    pdata = pseudo[dt]
    sdata = split[dt]
    lambdas_s = []
    lambdas_s_e = []
    lambdas_p = []
    lambdas_p_e = []
    lambdas_f = []
    lambdas_f_e = []
    ppp = [float(p) for p in pps]
    for p in pps:
        f_e, f_d = get_slopes(fdata[p], float(p), vector)
        p_e, p_d = get_slopes(pdata[p], float(p), vector)
        s_e, s_d = get_slopes(sdata[p], float(p), vector)
        lambdas_p.append(-p_d)
        lambdas_p_e.append(-p_e)
        lambdas_f.append(-f_d)
        lambdas_f_e.append(-f_e)
        lambdas_s.append(-s_d)
        lambdas_s_e.append(-s_e)

    lambda_t = [-ltp(l, float(dt)) for l in lambda_t_p]
    axs[i].set_title(rf"$\tau = {dt}$")
    label = "BE" if i == 0 else None
    axs[i].plot(
        ppp,
        lambdas_f,
        color="r",
        linestyle="-",
        marker="o",
        label=label,
        **kwargs,
    )
    axs[i].set_xlabel("$p$")
    label = "Pseudo" if i == 0 else None
    axs[i].plot(
        ppp,
        lambdas_p,
        color="b",
        linestyle="-",
        marker="^",
        label=label,
        **kwargs,
    )
    label = "Split" if i == 0 else None
    axs[i].plot(
        ppp,
        lambdas_s,
        color="g",
        linestyle="-",
        marker="d",
        label=label,
        **kwargs,
    )
    label = "Theoretical" if i == 0 else None
    for l_t_p, l_t in zip(lambda_t_p, lambda_t):
        axs[i].plot(
            l_t_p,
            l_t,
            color="k",
            linestyle="--",
            label=label,
        )
        label = None
    upper = max(upper, 1)  # XXX

for i, dt in enumerate(dts):
    axs[i].set_ybound(lower=0, upper=upper)
    axs[i].set_xbound(lower=0.5, upper=6.5)
    axs[i].set_xticks([1, 2, 4, 6])
fig.legend(loc="upper center", ncol=4, frameon=False)
fig.tight_layout()
fig.subplots_adjust(top=0.8, bottom=0.15)
if save:
    case = "scalar" if vector is False else "vector"
    fig.savefig(f"scripts/figures/tconv_lambda_{case}.png")
fig.show()
plt.show()
