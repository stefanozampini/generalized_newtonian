from firedrake import *

from ufl import grad, inner, div, conditional, gt
from functools import partial
from ufl_funcs import (
    Grad,
    norm2,
    u_energy,
    c_energy,
    absdiff,
    l2norm,
    h1snorm,
    modulus,
    average,
    qnormerr,
)
import numpy as np


def _constant(vector, v, x):
    if vector:
        return Constant((v,) * len(x))
    return Constant(v)


def _zero(vector, x):
    return _constant(vector, 0.0, x)


def _F_stress(f, p, v):
    if p is None:
        return inner(f, Grad(v)) * dx
    return (inner(f, Grad(v)) - inner(p, div(v))) * dx


def _F(f, p, v):
    if p is None:
        return inner(f, v) * dx
    return inner(f + grad(p), v) * dx


def _make_F(u, mu, p, f, fstress=False):
    if f is None:
        # mu(r^2)r is 0 when r^2 is zero
        g2 = norm2(Grad(u))
        f = conditional(gt(g2, 0), mu(g2) * Grad(u), 0 * Grad(u))
        if fstress:
            return partial(_F_stress, f, p)
        return partial(_F, -div(f), p)
    else:
        if fstress:
            return partial(_F_stress, f, p)
        return partial(_F, f, p)


class Solver:
    def __init__(self, msh, testcase, fem, sol, output):
        self.msh = msh
        self.testcase = testcase
        self.fem = fem
        self.sol = sol
        self.output = output

        self.log = []
        self.logd = []

        self.dt = Constant(self.sol.get("dt"))
        self.odt = float(self.dt)
        self.max_steps = self.sol.get("max_steps")
        self.atol = self.sol.get("atol")
        self.direct = self.sol.get("direct")
        self.direct_reinit = self.sol.get("direct_reinit")
        self.direct_use_objective = self.sol.get("direct_use_objective")
        self.split = self.sol.get("split")
        self.full_step = self.sol.get("full_step")
        self.adaptive_dt = self.sol.get("adaptive_dt")
        self.c0_random = self.sol.get("c0_random")

        self.init_problem()

    def make_zero_average(self, p):
        a = average(p)
        p.interpolate(p - a / self.volume)

    def init_problem(self, init_h=None):
        self.x = SpatialCoordinate(self.msh)
        self.n = FacetNormal(self.msh)
        self.volume = assemble(Constant(1) * dx(self.msh))

        self.vector = self.testcase.get("vector")
        self.exact = self.testcase.get("u_exact") is not None
        self.mu = self.testcase.get("mu")
        self.mu_prime = self.testcase.get("mu_prime")
        self.Phi = self.testcase.get("Phi")
        self.psi = self.testcase.get("psi")

        # Function spaces
        simplex = self.msh.topology_dm.isSimplex()
        dg = "DG" if simplex else "DQ"
        c_deg = self.fem.get("c_deg")
        u_deg = self.fem.get("u_deg")
        if c_deg < 0:
            if simplex:
                c_deg = 2 * (u_deg - 1)
            else:
                c_deg = 2 * u_deg
        self.C = FunctionSpace(self.msh, dg, c_deg, name="Ch")
        if self.vector:
            p_deg = u_deg - 1
            self.U = VectorFunctionSpace(self.msh, "CG", u_deg, name="Uh")
            scottvogelius = False
            if scottvogelius:
                self.Q = FunctionSpace(self.msh, dg, p_deg, name="Ph")
            else:
                self.Q = FunctionSpace(self.msh, "CG", p_deg, name="Ph")
            self.Z = self.U * self.Q
            self.V = self.C * self.Z
        else:
            cr = False
            if cr:
                self.U = FunctionSpace(self.msh, "CR", u_deg, name="Uh")
            else:
                self.U = FunctionSpace(self.msh, "CG", u_deg, name="Uh")
            self.V = self.C * self.U

        self.v_h = Function(self.V)
        self.v_h_prev = Function(self.V)
        if self.vector:
            self.z_h = Function(self.Z)
            self.c_h, self.u_h, self.p_h = self.v_h.subfunctions
        else:
            self.c_h, self.u_h = self.v_h.subfunctions
            self.p_h = None
        self.c_h.rename("c")
        self.u_h.rename("u")
        if self.vector:
            self.p_h.rename("p")
        self.cuerr_h = Function(self.C, name="cuerr")
        self.c_exact_h = Function(self.C, name="cexact_h")
        self.u_exact_h = Function(self.U, name="uexact_h")

        # Exact solution and rhs
        u_exact = self.testcase.get("u_exact")
        f_exact = self.testcase.get("f_exact")
        fstress = self.testcase.get("fstress")
        if f_exact is not None:
            f_exact = f_exact(self.x)
        if self.exact:
            self.diff_ex_h = Function(self.U, name="diff_exact_u")
            self.diff_exc_h = Function(self.C, name="diff_exact_c")
            if self.vector:
                u_exact, p_exact = u_exact
                pavg = average(p_exact(self.x))
                self.p_exact = p_exact(self.x) - pavg / self.volume
                self.diff_ex_p_h = Function(self.Q, name="diff_exact_p")
            else:
                self.p_exact = None

            self.u_exact = u_exact(self.x)
            self.u_exact_h.interpolate(self.u_exact)
            self.c_exact = norm2(Grad(self.u_exact))
            if self.vector:
                self.F = _make_F(
                    self.u_exact,
                    self.mu,
                    self.p_exact,
                    f_exact,
                    fstress=fstress,
                )
            else:
                self.F = _make_F(self.u_exact, self.mu, None, f_exact, fstress=fstress)
        else:
            self.u_exact = partial(_zero, self.vector)(self.x)
            self.c_exact = partial(_zero, False)(self.x)
            self.p_exact = partial(_zero, False)(self.x)
            if f_exact is not None:
                self.F = _make_F(
                    None,
                    None,
                    None,
                    f_exact,
                    fstress=fstress,
                )
            else:
                if self.vector:
                    self.F = lambda v: inner(Constant((1,) * len(self.x)), v) * dx
                else:
                    self.F = lambda v: inner(Constant(1), v) * dx

        # psi / Phi
        q_deg = self.fem.get("q_deg")
        if q_deg < 0:
            q_deg = 50
        self.psiq = False
        if hasattr(self.Phi, "setup"):
            self.Phi.setup(self.msh, q_deg)
        if hasattr(self.psi, "setup"):
            self.psi.setup(self.msh, q_deg)
            self.psiq = True
            self.psic_ex = Function(self.psi.q)
            self.psic_ex.assign(self.psi(self.c_exact))
            self.psic_ex_h = Function(self.psi.q)
            self.psic_ex_h.assign(self.psi(self.c_exact_h))
            self.psic = Function(self.psi.q)
        self.c_exact_h.interpolate(norm2(Grad(self.u_exact_h)))
        if self.exact:
            self.E_exact = u_energy(self.u_exact, self.F, self.Phi)
            self.Ec_exact = c_energy(
                self.u_exact, self.p_exact, self.F, self.c_exact, self.Phi, self.mu
            )
        else:
            self.E_exact = np.inf
            self.Ec_exact = np.inf

        self.setup_constraint()
        self.setup_step()
        self.setup_direct()
        self.initialize_fields(init_h=init_h)

    def initialize_fields(self, init_h=None):
        if self.c0_random:
            pcg = PCG64(seed=123456789)
            rg = RandomGenerator(pcg)
            c0 = rg.beta(self.C, 1.0, 2.0)
        else:
            c0 = Constant(1.0)
        self.c_h.interpolate(c0)
        self.solve_constraint()
        self.v_h_prev.assign(self.v_h)

    def setup_constraint(self):
        if self.vector:
            u, p = TrialFunctions(self.Z)
            v, q = TestFunctions(self.Z)
            a = (
                inner(self.mu(self.c_h) * Grad(u), Grad(v))
                - inner(p, div(v))
                - inner(div(u), q)
            ) * dx
            bc = DirichletBC(self.Z.sub(0), self.u_exact, "on_boundary")
            problem = LinearVariationalProblem(a, self.F(v), self.z_h, bcs=bc)
            nullspace = MixedVectorSpaceBasis(
                self.Z,
                [self.Z.sub(0), VectorSpaceBasis(constant=True, comm=self.Z.comm)],
            )
        else:
            phi = TestFunction(self.U)
            psi = TrialFunction(self.U)
            a = inner(self.mu(self.c_h) * Grad(psi), Grad(phi)) * dx
            bc = DirichletBC(self.U, self.u_exact, "on_boundary")
            problem = LinearVariationalProblem(a, self.F(phi), self.u_h, bcs=bc)
            nullspace = None

        solver_parameters = {
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": 500,
        }
        self.constraint_solver = LinearVariationalSolver(
            problem,
            nullspace=nullspace,
            transpose_nullspace=nullspace,  # RHS
            options_prefix="constraint_",
            solver_parameters=solver_parameters,
        )

    def monitor_direct(self, snes, it, fnorm):
        u_h = self.u_h
        if self.vector:
            u_h, _ = self.z_h.subfunctions
        u_en = u_energy(u_h, self.F, self.Phi)
        logprint = f"stepd {it}: EN_U {u_en:.3e}"
        if self.exact:
            logprint += f", EN_U_D {u_en - self.E_exact:.3e}"
        logprint += f", FNORMD {fnorm:.3e}"
        logd = (u_en, fnorm)
        self.logd.append(logd)
        PETSc.Sys.Print(logprint, flush=True)

    def setup_direct(self):
        en_kwargs = {}
        if self.vector:
            phi, tau = TestFunctions(self.Z)
            u, p = split(self.z_h)
            F = (
                inner(self.mu(norm2(Grad(u))) * Grad(u), Grad(phi))
                - inner(p, div(phi))
                - inner(div(u), tau)
            ) * dx - self.F(phi)
            bc = DirichletBC(self.Z.sub(0), self.u_exact, "on_boundary")
            if self.direct_use_objective:
                energy = self.Phi(norm2(Grad(u))) * dx - self.F(u)
                en_kwargs = {"objective": energy}
            problem = NonlinearVariationalProblem(F, self.z_h, bcs=bc, **en_kwargs)
            nullspace = MixedVectorSpaceBasis(
                self.Z,
                [self.Z.sub(0), VectorSpaceBasis(constant=True, comm=self.Z.comm)],
            )
        else:
            phi = TestFunction(self.U)
            u = self.u_h
            # mu(r^2)r is 0 when r^2 is zero
            F = inner(
                conditional(
                    gt(norm2(Grad(u)), 0),
                    self.mu(norm2(Grad(u))) * Grad(u),
                    0 * Grad(u),
                ),
                Grad(phi),
            ) * dx - self.F(phi)
            bc = DirichletBC(self.U, self.u_exact, "on_boundary")
            if self.direct_use_objective:
                energy = self.Phi(norm2(Grad(u))) * dx - self.F(u)
                en_kwargs = {"objective": energy}
            problem = NonlinearVariationalProblem(F, self.u_h, bcs=bc, **en_kwargs)
            nullspace = None

        solver_parameters = {
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": 500,
            "snes_linesearch_type": "bt",
            "snes_atol": self.atol,
            "snes_rtol": 1.0e-50,
            "snes_stol": 0,
            "snes_max_it": 200,
        }
        self.direct_solver = NonlinearVariationalSolver(
            problem,
            nullspace=nullspace,
            transpose_nullspace=nullspace,  # RHS
            options_prefix="direct_",
            solver_parameters=solver_parameters,
        )
        self.direct_solver.snes.setMonitor(self.monitor_direct)

    def setup_step(self, final=False):
        nullspace = None
        c_prev, *_ = self.v_h_prev.subfunctions
        if self.vector:
            psi, phi, tau = TestFunctions(self.V)
            c, u, p = split(self.v_h)
        else:
            psi, phi = TestFunctions(self.V)
            c, u = split(self.v_h)
        F = inner(self.mu(c) * Grad(u), Grad(phi)) * dx - self.F(phi)
        if self.vector:
            F += (-inner(p, div(phi)) - inner(div(u), tau)) * dx
            nullspace = MixedVectorSpaceBasis(
                self.V,
                [
                    self.V.sub(0),
                    self.V.sub(1),
                    VectorSpaceBasis(constant=True, comm=self.V.comm),
                ],
            )
        F += (
            inner(
                (c - c_prev) / self.dt - norm2(Grad(u)) + c,
                psi,
            )
            * dx
        )
        bc = DirichletBC(self.V.sub(1), self.u_exact, "on_boundary")
        problem = NonlinearVariationalProblem(F, self.v_h, bcs=bc)
        solver_parameters = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": 500,
            "snes_linesearch_type": "bt",
            "snes_linesearch_order": "1",  # just backtrack until mu(c) is positive
            "snes_atol": 1.0e-14,
        }
        if not self.full_step:  # Pseudo iteration
            solver_parameters.update(
                {
                    "snes_norm_schedule": "finalonly",
                    "snes_max_it": 1,
                }
            )
        self.step_solver = NonlinearVariationalSolver(
            problem,
            nullspace=nullspace,
            transpose_nullspace=nullspace,  # RHS
            options_prefix="step_",
            solver_parameters=solver_parameters,
        )

    def logstep(self, cnt, fnorm, fnormd, direct=False):
        dt = float(self.dt)
        u_en = u_energy(self.u_h, self.F, self.Phi)
        c_en = c_energy(self.u_h, self.p_h, self.F, self.c_h, self.Phi, self.mu)
        dmerr = self.eval_dmetric(self.c_h, self.u_h)

        outs = (dt, u_en, c_en, dmerr)
        logprint = f"step {cnt}: DT {dt:.3e}, EN_U {u_en:.3e}, EN_C {c_en:.3e}, EN_D = {u_en - c_en:.3e}, ERR_DM {dmerr:.3e}"
        if self.exact:
            logprint = (
                logprint
                + f", EN_U_D {u_en - self.E_exact:.3e}, EN_C_D {c_en - self.Ec_exact:.3e}"
            )
            diff_ex_f = absdiff(self.u_exact, self.u_h)
            l2err = l2norm(modulus(diff_ex_f))
            h1err = h1snorm(diff_ex_f)
            outs += (l2err, h1err)
            logprint = logprint + f", ERR_L2 {l2err:.3e}, ERR_H1 {h1err:.3e}"
            if self.vector:
                diff_ex_p = absdiff(self.p_exact, self.p_h)
                l2err = l2norm(modulus(diff_ex_p))
                outs += (l2err,)
                logprint = logprint + f", ERR_L2p {l2err:.3e}"
            if self.psiq:
                self.psic.assign(self.psi(self.c_h))
                psicerr = l2norm(self.psic_ex - self.psic)
            else:
                psicerr = l2norm(self.psi(self.c_h) - self.psi(self.c_exact))
            qnerr = qnormerr(self.mu, self.u_exact, self.u_h)
            outs += (
                psicerr,
                qnerr,
            )
            logprint = logprint + f", ERR_PSI {psicerr:.3e}, ERR_QN {qnerr:.3e}"

        outs += (fnorm, fnormd)
        logprint = logprint + f", FNORM {fnorm:.3e}, FNORMD {fnormd:.3e}"
        PETSc.Sys.Print(logprint)
        if not direct:
            self.log.append(outs)

        if self.output is None:
            return
        diff_c_f = absdiff(norm2(Grad(self.u_h)), self.c_h)
        self.cuerr_h.interpolate(diff_c_f)
        if self.vector:
            outs = (self.c_h, self.u_h, self.p_h, self.cuerr_h)
        else:
            outs = (self.c_h, self.u_h, self.cuerr_h)
        if self.exact:
            self.diff_ex_h.interpolate(diff_ex_f)
            outs += (self.diff_ex_h,)
            self.diff_exc_h.interpolate(absdiff(self.c_h, self.c_exact))
            outs += (self.diff_exc_h,)
            if self.vector:
                self.diff_ex_p_h.interpolate(diff_ex_p)
                outs += (self.diff_ex_p_h,)

        residual = self.compute_residual_norm(direct=True, rrhs=True)
        residual.rename("residual")
        outs += (residual,)
        self.output.write(*outs, time=cnt)

    def get_log_data(self, direct=False):
        if direct:
            out = np.array(self.logd)
            names = ["u_en", "fnormd"]
            return out, names
        out = np.array(self.log)
        names = ["dt", "u_en", "c_en", "dm_err"]
        if self.exact:
            names += ["L2", "H1"]
            if self.vector:
                names += ["L2p"]
            names += ["PSI", "QNORM"]
        names += ["fnorm", "fnormd"]
        return out, names

    def converged(self, cnt, fnorm):
        conv = False
        return conv | (cnt >= self.max_steps) | (fnorm < self.atol)

    def solve_constraint(self):
        if self.vector:
            u, p = self.z_h.subfunctions
            u.assign(self.u_h)
            p.assign(self.p_h)
        self.constraint_solver.solve()
        if self.vector:
            u, p = self.z_h.subfunctions
            self.u_h.assign(u)
            self.p_h.assign(p)
            self.make_zero_average(self.p_h)

    def eval_distance(self, c1, c2):
        if self.psiq:
            psic1 = self.psi.eval(c1)
            psic2 = self.psi.eval(c2)
            d = l2norm(psic1 - psic2)
        else:
            d = l2norm(self.psi(c1) - self.psi(c2))
        return d

    def eval_dmetric(self, c, u):
        d = assemble((abs(self.mu_prime(c)) / 2 * (c - norm2(Grad(u))) ** 2) * dx)
        return d ** (1 / 2)

    def compute_residual_norm(self, v=None, direct=False, rrhs=False):
        odt = float(self.dt)
        self.dt.assign(1.0e16)
        if direct:
            problem = self.direct_solver._problem
        else:
            problem = self.step_solver._problem
        F = problem.F
        bcs = problem.bcs
        u = problem.u
        if v is not None:
            su = Function(u)
            u.assign(v)
        elif direct and self.vector:
            u, p = u.subfunctions
            u.assign(self.u_h)
            p.assign(self.p_h)
        rhs = assemble(F, bcs=bcs)
        if v is not None:
            u.assign(su)
        self.dt.assign(odt)
        if rrhs:
            return rhs
        norms = []
        with rhs.dat.vec_ro as pf:
            norms.append(pf.norm())
        if not direct:
            for f in rhs.subfunctions:
                with f.dat.vec_ro as pf:
                    norms.append(pf.norm())
        return norms

    def solve(self):
        cnt = 0
        fnorm = self.compute_residual_norm()[0]
        fnormd = self.compute_residual_norm(direct=True)[
            0
        ]  # residual norm of the direct FEM problem
        self.logstep(cnt, fnorm, fnormd)
        while not self.converged(cnt, fnormd):
            # n0 = self.compute_residual_norm()
            # n0d = self.compute_residual_norm(direct=True)
            self.v_h_prev.assign(self.v_h)
            u_en_0 = u_energy(self.u_h, self.F, self.Phi)
            # c_en_0 = c_energy(self.u_h, self.p_h, self.F, self.c_h, self.Phi, self.mu)
            self.c_h.interpolate(
                (self.c_h + self.dt * norm2(Grad(self.u_h))) / (1 + self.dt)
            )
            # u_en_1 = u_energy(self.u_h, self.F, self.Phi)
            # c_en_1 = c_energy(self.u_h, self.p_h, self.F, self.c_h, self.Phi, self.mu)
            n1 = self.compute_residual_norm()
            # n1d = self.compute_residual_norm(direct=True)
            failsolve = False
            if self.split:  # split iteration
                self.solve_constraint()
            else:  # either BE or Pseudo, depending on self.full_step
                try:
                    self.step_solver.solve()
                except ConvergenceError as e:
                    failsolve = True
                    warning("STEP SOLVER CONVERGENCE ERROR\n'%s'" % e)
                if self.vector:
                    self.make_zero_average(self.p_h)

            n2 = self.compute_residual_norm()
            # n2d = self.compute_residual_norm(direct=True)
            u_en_2 = u_energy(self.u_h, self.F, self.Phi)
            c_en_2 = c_energy(self.u_h, self.p_h, self.F, self.c_h, self.Phi, self.mu)

            if self.adaptive_dt:
                t1 = u_en_2 > u_en_0 + 0.0001
                t2 = n1[1] / n2[1] < 1
                if failsolve or (t1 and t2):
                    PETSc.Sys.Print(
                        " ------ FAIL:",
                        u_en_2 - u_en_0,
                        n1[1] / n2[1],
                    )
                    s_fail = 0.1
                    self.dt.assign(float(self.dt) * s_fail)
                    self.v_h.assign(self.v_h_prev)
                else:
                    # factor depending on full norm or constraint error
                    which = 0 if self.adaptive_dt == 1 else 2
                    fact = 1
                    maxfact = 2 if self.adaptive_dt == 1 else 1000000
                    if n1[1] / n2[1] < 1:  # or fnorm < self.atol:
                        fact = min(n1[which] / n2[which], maxfact)
                    self.dt.assign(float(self.dt) * fact)
                    fnorm = n2[0]
            else:
                fnorm = n2[0]
            fnormd = self.compute_residual_norm(direct=True)[0]

            cnt += 1
            self.logstep(cnt, fnorm, fnormd)

        # Solve the Euler-Lagrange equations of the original problem
        if self.direct:
            if self.direct_reinit:
                self.initialize_fields()
            try:
                if self.vector:
                    u, p = self.z_h.subfunctions
                    u.assign(self.u_h)
                    p.assign(self.p_h)
                self.direct_solver.solve()
            except ConvergenceError as e:
                warning("DIRECT SOLVER CONVERGENCE ERROR\n'%s'" % e)
            if self.vector:
                u, p = self.z_h.subfunctions
                self.u_h.assign(u)
                self.p_h.assign(p)
                self.make_zero_average(self.p_h)
            self.c_h.interpolate(norm2(Grad(self.u_h)))
            fnorm = self.compute_residual_norm()[0]
            fnormd = self.compute_residual_norm(direct=True)[0]
            cnt += 1
            self.logstep(cnt, fnorm, fnormd, direct=True)
