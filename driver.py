import petsc4py, sys

petsc4py.init(sys.argv)

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", message="Calling petsc4py.init but PETSc has already been initialized"
    )
    from firedrake import VTKFile

from petsc4py import PETSc
from testcases import make_test_case
from mesh_utils import make_mesh
from utils import pickle_save
import numpy as np
import scipy
from solver import Solver

print = PETSc.Sys.Print

# Read command-line options
opts = PETSc.Options()

# FEM
u_deg = opts.getInt("u_deg", 1)  # degree k of FEM
c_deg = opts.getInt("c_deg", -1)  # automatic
q_deg = opts.getInt("q_deg", -1)  # automatic
fem = {
    "u_deg": u_deg,
    "c_deg": c_deg,
    "q_deg": q_deg,
}

# mesh
msh = make_mesh(opts)

# model
mms = opts.getInt("mms", 0)  # mms to choose from, see testcases.py
model = opts.getString("model", "powerlaw")  # model to use, see testcases.py
vector = opts.getBool("vector", False)  # activate incompressible vector-valued model
fstress = opts.getBool("fstress", True)  # forcing term testing against gradients
u_exact, f_exact, mu, mu_prime, Phi, psi = make_test_case(
    model, mms, vector, opts, fstress
)
testcase = {
    "u_exact": u_exact,
    "f_exact": f_exact,
    "mu": mu,
    "mu_prime": mu_prime,
    "Phi": Phi,
    "psi": psi,
    "vector": vector,
    "fstress": fstress,
}

# solver
dt = opts.getReal("dt", 1.0)  # tau_0
adaptive_dt = opts.getInt("adaptive_dt", 0)  # use adaptive time stepping
max_steps = opts.getReal("max_steps", 10)  # number of steps
atol = opts.getReal("atol", 0)  # absolute tolerance for stopping the iterations
direct = opts.getInt(
    "direct", 0
)  # solve the original nonlinear Galerkin problem (aka the direct method)
split = opts.getBool("split", False)  # use the split iteration
full_step = opts.getBool("full_step", False)  # use the BE iteration if True
c0_random = opts.getBool("c0_random", False)  # use random initial condition
sol = {
    "dt": dt,
    "adaptive_dt": adaptive_dt,
    "max_steps": max_steps,
    "direct": direct,
    "split": split,
    "full_step": full_step,
    "atol": atol,
    "c0_random": c0_random,
}

# Output file
output = None
outfile = opts.hasName("outfile")
if outfile:
    outfile = opts.getString("outfile", "output.pvd")
    output = VTKFile(outfile)

solver = Solver(msh, testcase, fem, sol, output)
solver.solve()

plot = opts.getBool("plot", False)
logfile = opts.getString("logfile", "")
if PETSc.COMM_WORLD.rank == 0 and (plot or logfile):
    data, names = solver.get_log_data()
    ddata, dnames = solver.get_log_data(direct=True)
    if logfile:
        pdata = {
            "problem": {
                "model": model,
                "case": mms,
                "vector": vector,
                "fstress": fstress,
            },
            "fem": fem,
            "sol": sol,
            "exact": solver.exact,
            "E_exact": solver.E_exact,
            "Ec_exact": solver.Ec_exact,
        }
        out_dict = {
            "edata": data,
            "edata_n": names,
            "ddata": ddata,
            "ddata_n": dnames,
            "pdata": pdata,
        }
        pickle_save(logfile, out_dict)

    if plot:
        import matplotlib.pyplot as plt
        import math

        fig, ax = plt.subplots(1, 1, num="energies")
        axt = ax.twinx()
        ax.plot(data[:, 1], marker=".", label=names[1], color="r")
        ax.plot(data[:, 2], marker=".", label=names[2], color="b")
        axt.semilogy(
            abs(data[:, 1] - data[:, 2]), linestyle="--", label="diff", color="g"
        )
        fig.legend()

        plt.figure("errors")
        est = 3
        een = 8 + int(vector)

        tt = np.concatenate([[0], np.cumsum(data[:-1, 0])])
        for ee, c in zip(range(est, een), ["r", "g", "b", "k", "y", "m", "o"]):
            s = 2 * data.shape[0] // 5
            e = 3 * data.shape[0] // 5
            if s == e:
                s = 0
                e = data.shape[0]
            pf = np.polyfit(tt[s:e], np.log(data[s:e, ee]), 1)
            plt.plot(
                tt,
                data[:, ee],
                ".",
                label=names[ee] + f" (fit {pf[0]:.3e})",
                color=c,
            )
            plt.plot(
                tt[s:e],
                100 * np.exp(pf[0] * tt[s:e] + pf[1]),
                linestyle="--",
                color=c,
            )
            plt.yscale("log")

        plt.legend()
        plt.show()
