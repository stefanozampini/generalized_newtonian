import ufl
from firedrake import assemble, dx


def clip_expr(c, m):
    return ufl.conditional(ufl.ge(c, m), c, m)


def norm2(u):
    return ufl.inner(u, u)


def modulus(u):
    return ufl.sqrt(norm2(u))


def Grad(u):
    if hasattr(u, "ufl_shape") and len(u.ufl_shape):
        return ufl.sym(ufl.grad(u))
    return ufl.grad(u)


def average(u):
    return assemble(u * dx)


def lpnorm(u, p):
    n = assemble((ufl.inner(u, u) ** (p / 2)) * dx)
    return n ** (1 / p)


def wpsnorm(u, p):
    return lpnorm(ufl.grad(u), p)


def l2norm(u):
    return lpnorm(u, 2)


def h1snorm(u):
    return wpsnorm(u, 2)


def qnormerr(mu, u1, u2):
    def _V(mu, Du):
        return ufl.sqrt(mu(norm2(Du))) * Du

    return l2norm(_V(mu, Grad(u1)) - _V(mu, Grad(u2)))


def absdiff(u1, u2):
    if hasattr(u1, "ufl_shape") and len(u1.ufl_shape):
        t = [abs(t1 - t2) for t1, t2 in zip(u1, u2)]
        return ufl.as_vector(t)
    return abs(u1 - u2)


def u_energy(u, F, Phi):
    return assemble(Phi(norm2(Grad(u))) * dx - F(u))


def c_energy(u, p, F, c, Phi, mu):
    c = clip_expr(c, 1.0e-16)  # XXX high-order
    return assemble(
        (Phi(c) - 0.5 * mu(c) * c + 0.5 * mu(c) * norm2(Grad(u))) * dx - F(u)
    )
