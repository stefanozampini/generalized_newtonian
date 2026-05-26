import ufl
from functools import partial
from math import sqrt
from firedrake import Function, FunctionSpace
from scipy.special import hyp2f1
from ufl_funcs import clip_expr


def _powerlaw_mu(p, t):
    return t ** ((p - 2) / 2)


def _powerlaw_mu_prime(p, t):
    return ((p - 2) / 2) * t ** ((p - 4) / 2)


def _powerlaw_Phi(p, t):
    return t ** (p / 2) / p


def _powerlaw_psi(p, t):
    if p == 2:
        C = 1
    else:
        C = 2 / p * sqrt(abs(p - 2))
    return C * clip_expr(t, 1.0e-16) ** (p / 4)  # XXX


def _carreau_yasuda_mu(mu_inf, mu_0, lam, a, n, t):
    return mu_inf + (mu_0 - mu_inf) * (1 + (lam**a) * (t ** (a / 2))) ** ((n - 1) / a)


def _carreau_yasuda_mu_prime(mu_inf, mu_0, lam, a, n, t):
    x = (lam**a) * (t ** (a / 2))
    q = (n - 1) / a
    K = (mu_0 - mu_inf) * (n - 1) / 2
    return K * x * (1 + x) ** (q - 1) / t


def _carreau_yasuda_Phi(mu_inf, mu_0, lam, a, n, t):
    ha = (1 - n) / a
    hb = 2 / a
    hc = 1 + 2 / a
    ht = -(lam**a) * (t**a / 2)
    v = t / 2 * (mu_inf + (mu_0 - mu_inf) * hyp2f1(ha, hb, hc, ht))
    return v


def _carreau_yasuda_psi(mu_inf, mu_0, lam, a, n, t):
    ha = (a + 1 - n) / (2 * a)
    hb = (a + 2) / (2 * a)
    hc = (3 * a + 2) / (2 * a)
    ht = -(lam**a) * (t**a / 2)
    dmu = mu_0 - mu_inf
    factor = (
        (4 * lam ** (a / 2))
        / (a + 2)
        * t ** ((a + 2) / 4)
        * sqrt((dmu * abs(n - 1)) / 2)
    )
    v = factor * hyp2f1(ha, hb, hc, ht)
    return v


def _optimal_mu(m_1, m_2, e_1, e_2, t):
    return ufl.conditional(
        ufl.lt(t, e_1**2),
        m_2,
        ufl.conditional(ufl.lt(t, e_2**2), e_1 * m_2 / ufl.sqrt(t), m_1),
    )


def _optimal_mu_prime(m_1, m_2, e_1, e_2, t):
    return  # TODO


def _optimal_Phi(m_1, m_2, e_1, e_2, t):
    return ufl.conditional(
        ufl.lt(t, e_1**2),
        t * m_2 / 2,
        ufl.conditional(
            ufl.lt(t, e_2**2),
            e_1 * m_2 * (ufl.sqrt(t) - e_1 / 2),
            t * m_1 / 2 - e_1 * m_2 * (e_1 - e_2) / 2,
        ),
    )


def _bingham_mu(nu, sigma, eps, t):
    return 2 * nu + sigma / (ufl.sqrt(t + eps**2))


def _bingham_mu_prime(nu, sigma, eps, t):
    return  # TODO


def _bingham_Phi(nu, sigma, eps, t):
    return nu * t + sigma * (ufl.sqrt(t + eps**2) - eps)


# class to evaluete Quadrature functions
class FEval:
    def __init__(self, F):
        self.F = F

    def setup(self, mesh, degree):
        self.q = Function(FunctionSpace(mesh, "Quadrature", degree))

    def __call__(self, u):
        self.q.interpolate(u)
        self.q.dat.data[:] = self.F(self.q.dat.data)
        return self.q

    def eval(self, u):
        q = Function(self.q)
        q.interpolate(u)
        q.dat.data[:] = self.F(q.dat.data)
        return q


def _divfree(A, x):
    if len(x) == 2:
        return ufl.as_vector([A(x).dx(1), -A(x).dx(0)])
    else:
        raise NotImplementedError(f"DIM {len(x)}")


def _fzero(t):
    return t


def _zero(x):
    return x[0] - x[0]


def make_test_case(model, mms, vector, opts, fstress):
    Phi_q = False
    psi = _fzero
    psi_q = False
    mu_prime = _fzero
    if model == "powerlaw":
        p = opts.getReal("powerlaw_p", 1.5)
        mu = partial(_powerlaw_mu, p)
        mu_prime = partial(_powerlaw_mu_prime, p)
        Phi = partial(_powerlaw_Phi, p)
        psi = partial(_powerlaw_psi, p)
    elif model == "carreau_yasuda":
        n = opts.getReal("carreau_yasuda_n", 0.356)
        mu_inf = opts.getReal("carreau_yasuda_muinf", 0.0035)
        mu_0 = opts.getReal("carreau_yasuda_mu0", 0.056)
        lam = opts.getReal("carreau_yasuda_lambda", 3.3)
        a = opts.getReal("carreau_yasuda_a", 2.0)
        mu = partial(_carreau_yasuda_mu, mu_inf, mu_0, lam, a, n)
        Phi = partial(_carreau_yasuda_Phi, mu_inf, mu_0, lam, a, n)
        mu_prime = partial(_carreau_yasuda_mu_prime, mu_inf, mu_0, lam, a, n)
        psi = partial(_carreau_yasuda_psi, mu_inf, mu_0, lam, a, n)
        Phi_q = True
        psi_q = True
    elif model == "optimal":
        lam = opts.getReal("optimal_lambda", 0.0084)
        m_1 = opts.getReal("optimal_m1", 1)
        m_2 = opts.getReal("optimal_m2", 2)
        e_1 = opts.getReal("optimal_e1", sqrt(2 * lam * m_1 / m_2))
        e_2 = opts.getReal("optimal_e2", m_2 * e_1 / m_1)
        mu = partial(_optimal_mu, m_1, m_2, e_1, e_2)
        Phi = partial(_optimal_Phi, m_1, m_2, e_1, e_2)
    elif model == "bingham":
        b_nu = opts.getReal("bingham_nu", 1)
        b_sigma = opts.getReal("bingham_sigma", 0.3)
        b_eps = opts.getReal("bingham_eps", 1.0e-1)
        mu = partial(_bingham_mu, b_nu, b_sigma, b_eps)
        Phi = partial(_bingham_Phi, b_nu, b_sigma, b_eps)
    else:
        raise NotImplementedError(f"Unknown model {model}")

    u_exact = None
    f_exact = None
    if mms == 0:
        pass
    elif mms == 1:  # trig
        kx = opts.getReal("mms_trig_kx", 0.5)
        ky = opts.getReal("mms_trig_ky", 0.5)

        def _trig(k, x):
            kx, ky = k
            kkx = 2 * kx * ufl.pi
            kky = 2 * ky * ufl.pi
            return ufl.sin(kkx * x[0]) * ufl.sin(kky * x[1])

        if vector:
            p_exact = partial(_trig, (kx, ky))
            u_exact = partial(_divfree, p_exact)
            u_exact = (u_exact, p_exact)
        else:
            u_exact = partial(_trig, (kx, ky))
    elif mms == 2:  # radial
        alpha = 1.0
        shift = 0.0
        factor = 1.0

        bl1 = opts.getBool("mms_radial_bl1", False)
        needle = opts.getBool("mms_radial_needle", False)
        if bl1:
            if model != "powerlaw":
                raise RuntimeError("BL1 for powerlaw")
            sigma = opts.getReal("mms_radial_bl1_sigma", 0)
            alpha = (sigma + p) / (p - 1)
            shift = -1
            factor = -(p - 1) / (sigma + p) * (sigma + 2) ** (1 / (1 - p))
        elif needle:  # needle in Kacanov p<2 paper
            if model != "powerlaw":
                raise RuntimeError("NEEDLE for powerlaw")
            alpha = (p - 1) / p
            shift = -1
        else:
            alpha = opts.getReal("mms_radial_alpha", alpha)
            shift = opts.getReal("mms_radial_shift", shift)
            factor = opts.getReal("mms_radial_factor", factor)

        def _radial(f, s, a, x):
            r2 = sum([xx**2 for xx in x])
            return f * (r2 ** (a / 2) + s)

        if vector:
            p_exact = partial(_radial, factor, shift, alpha)
            u_exact = partial(_divfree, p_exact)
            u_exact = (u_exact, p_exact)
        else:
            u_exact = partial(_radial, factor, shift, alpha)
            if bl1 and not fstress:
                f_exact = partial(_radial, 1, 0, sigma)

    elif mms == 3:  # bump in Kacanov p<2 paper

        def _bump(x):
            u = 1
            for xx in x:
                u = u * (xx**2 - 1)
            return u

        if vector:
            p_exact = _bump
            u_exact = partial(_divfree, p_exact)
            u_exact = (u_exact, p_exact)
        else:
            u_exact = _bump
    elif mms == 4:  # p-Stokes for tconv

        def _uex(x):
            return 4 * (1 - x[0] ** 2 - x[1] ** 2) * ufl.as_vector([x[1], -x[0]])

        def _pex(x):
            return x[0] ** 2 + x[1] ** 2

        if vector:
            u_exact = (_uex, _pex)
        else:
            u_exact = _pex
    elif mms == 5:  # bingham

        def _uinf(x, vector=False):
            v0 = 0.16
            v1 = ufl.conditional(ufl.le(x[1], 0.2), (0.4 - 2 * x[1]) ** 2, 0)
            v2 = ufl.conditional(ufl.ge(x[1], 0.8), (2 * x[1] - 1.6) ** 2, 0)
            v = 0.125 * (v0 - v1 - v2)
            if vector:
                return ufl.as_vector([v, 0])
            return v

        if vector:
            u_exact = (partial(_uinf, vector=True), _zero)
        else:
            u_exact = _uinf

    elif mms == 6:  # Carreau-Yasuda
        a = opts.getReal("mms_c6_a", 0.01)
        g = opts.getReal("mms_c6_g", 2 / 1.5 - 1 + a)

        def _uex(a, x):
            r2 = sum([xx**2 for xx in x])
            return r2 ** (a / 2) * ufl.as_vector([x[1], -x[0]])

        def _pex(g, x):
            r2 = sum([xx**2 for xx in x])
            return -(r2 ** (g / 2))

        if vector:
            u_exact = (partial(_uex, a), partial(_pex, g))
        else:
            u_exact = partial(_pex, g)

    elif mms == 7:  # |grad u|^2 zero on a set of positive measure
        if vector:
            raise NotImplementedError(f"Unknown mms {mms}")
        alpha = 1
        beta = 1
        alpha = opts.getReal("mms_radialzero_alpha", alpha)
        beta = opts.getReal("mms_radialzero_beta", beta)
        a = opts.getReal("mms_radialzero_a", 1)
        b = opts.getReal("mms_radialzero_b", 0.6)
        shift = opts.getReal("mms_radialzero_shift", 0)

        def _radial_zero(s, a, b, alpha, beta, x):
            r = ufl.sqrt(sum([xx**2 for xx in x]))
            u = ufl.conditional(ufl.ge(r, b), 0, a * (1 - (r / b) ** alpha) ** beta)
            return u + s

        u_exact = partial(_radial_zero, shift, a, b, alpha, beta)
    else:
        raise NotImplementedError(f"Unknown mms {mms}")
    if Phi_q:
        Phi = FEval(Phi)
    if psi_q:
        psi = FEval(psi)
    return u_exact, f_exact, mu, mu_prime, Phi, psi
