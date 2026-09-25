import numpy as np
from numpy.fft import fftn, ifftn
import scipy 
from tqdm import tqdm

def vdot_z(z1, z2):
    return np.einsum("ijk,ijk->ij", np.conj(z1), z2)

def vouter_z(z1, z2):
    return np.einsum("ijk,ijl->ijkl", z1, np.conj(z2))

def U1_clip(x):
    return np.mod(x+np.pi, 2*np.pi) - np.pi

class CPN_RefVil_HMCSampler:
    """
    CP^{N-1} + U(1) + Z sampler with global constrained HMC updates.

    - z field: shape (Lx, Ly, N), complex, with per-site norm |z|^2 = 1.
    - U field: shape (Lx, Ly, 2), complex phases on links.
    - a field: real phases such that U = exp(1j * a).
    - s field: shape (Lx, Ly), integer field

    Action is improved form with beta1 * |z' z|^2 term or log I0(beta1 * |z' z|) term,
    determined by mod = 0, 1.

    Action is Villainized, with alpha / 2 * (da + 2 pi s)^2 and -alpha1 * cos(da) term.

    This sampler has s_max, so that s <= s_max
    """

    def __init__(
        self,
        N,
        Lx,
        Ly,
        beta,
        beta1,
        alpha,
        alpha1,
        seed=None,
        epsilon=0.02,
        n_leapfrog=50,
        mass_a=1.0,
        mass_z=1.0,
        s_step=0.5,
        s_update_num=1,
        s_max=100,
    ):
        self.Lx = Lx
        self.Ly = Ly
        if N <= 1:
            raise Exception("N too small")
        self.N = N
        self.beta = beta
        self.beta1 = beta1
        self.alpha = alpha
        self.alpha1 = alpha1
        self.V = Lx * Ly

        self.epsilon = float(epsilon)
        self.n_leapfrog = int(n_leapfrog)
        self.mass_a = float(mass_a)
        self.mass_z = float(mass_z)
        self.s_step = float(s_step)
        self.s_update_num = int(s_update_num)
        self.s_max = int(s_max)

        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.n_leapfrog <= 0:
            raise ValueError("n_leapfrog must be positive")
        if self.mass_a <= 0 or self.mass_z <= 0:
            raise ValueError("mass_a and mass_z must be positive")

        if seed is not None:
            np.random.seed(seed)

        self.z = self._random_unit_vectors((Lx, Ly, N))

        self.a = 2 * np.pi * np.random.rand(Lx, Ly, 2)
        self._wrap_phase_inplace()
        self._sync_U_from_a()
        # s: integer in every plaq
        self.s = np.zeros((Lx, Ly), dtype=int)

        self.accepted_hmc = 0
        self.attempted_hmc = 0
        self.accepted_metro = 0
        self.attempted_metro = 0

    @property
    def accept_rate(self):
        '''accteptance rate of HMC and metro updates.'''
        if self.attempted_hmc == 0:
            hmc_rate = 0.0
        else:
            hmc_rate = self.accepted_hmc / self.attempted_hmc
        if self.attempted_metro == 0:
            metro_rate = 0.0
        else:
            metro_rate = self.accepted_metro / self.attempted_metro
        return hmc_rate, metro_rate

    def _random_unit_vectors(self, shape):
        real = np.random.randn(*shape)
        imag = np.random.randn(*shape)
        v = real + 1j * imag
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / norm

    def _wrap_phase_inplace(self):
        self.a = (self.a + np.pi) % (2 * np.pi) - np.pi

    def _sync_U_from_a(self):
        self.U = np.exp(1j * self.a)

    def _normalize_z_inplace(self):
        norm = np.linalg.norm(self.z, axis=-1, keepdims=True)
        self.z /= norm

    def _project_tangent(self, z, field):
        """
        Project complex field onto tangent space of |z|^2=1 manifold.
        Constraint is Re(<z, field>) = 0 at each site.
        """
        overlap = np.einsum("ijk,ijk->ij", np.conj(z), field)
        return field - np.real(overlap)[:, :, None] * z

    def _spin_inner_x(self):
        return np.einsum("ijk,ijk->ij", np.conj(self.z), np.roll(self.z, -1, axis=0))

    def _spin_inner_y(self):
        return np.einsum("ijk,ijk->ij", np.conj(self.z), np.roll(self.z, -1, axis=1))

    def _plaquette_da(self):
        a = self.a
        f = a[:,:,0] + np.roll(a, -1, axis=0)[:,:,1] - np.roll(a, -1, axis=1)[:,:,0] - a[:,:,1] + 2 * np.pi * self.s
        return f

    def action_density(self, mod=0):
        z = self.z
        U = self.U
        V = self.V

        cp_term = np.real(U[:, :, 0] * np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=0)), z))
        cp_term += np.real(U[:, :, 1] * np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=1)), z))
        if mod == 0:
            cp_term_1 = np.abs(np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=0)), z)) ** 2 - 1
            cp_term_1 += np.abs(np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=1)), z)) ** 2 - 1
            cp_term_1 *= self.N * self.beta1
        elif mod == 1:
            cp_term_1 = np.sign(self.beta1) * np.log(scipy.special.i0(np.abs(np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=0)), z)) * 2 * self.N * self.beta1))
            cp_term_1 += np.sign(self.beta1) * np.log(scipy.special.i0(np.abs(np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=1)), z)) * 2 * self.N * self.beta1))
        plaq = self._plaquette_da()
        plaq_term = (plaq ** 2) / 2
        plaq_term_1 = np.cos(plaq)

        return -2.0 * self.N * self.beta * (cp_term - 2) - cp_term_1 + self.alpha * (plaq_term) - self.alpha1 * (plaq_term_1 - 1)

    def action(self, mod=0):
        return np.sum(self.action_density(mod=mod))

    def action_s(self, s, x, y):
        '''
        Local action of plaquette variable s.
        '''
        Lx, Ly = self.Lx, self.Ly
        a = self.a
        xp = (x+1) % Lx
        yp = (y+1) % Ly
        return self.alpha / 2 * (a[x, y, 0] + a[xp, y, 1] - a[x, yp, 0] - a[x, y, 1] + 2 * np.pi * s) ** 2

    def _z_force(self, mod=0):
        """
        Returns force F_z = -dS/d(conj(z)) * 2.
        """
        z = self.z
        U = self.U
        N = self.N
        beta = self.beta
        beta1 = self.beta1

        F = np.zeros_like(z)
        F1 = np.zeros_like(z)
        inner_x = self._spin_inner_x()
        inner_y = self._spin_inner_y()
        if self.Lx > 1:
            F += np.roll(U[:, :, 0], 1, axis=0)[:, :, None] * np.roll(z, 1, axis=0)
            F += np.conj(U[:, :, 0])[:, :, None] * np.roll(z, -1, axis=0)
            if mod == 0:
                F1 += np.roll(inner_x, 1, axis=0)[:, :, None] * np.roll(z, 1, axis=0) * beta1
                F1 += np.conj(inner_x)[:, :, None] * np.roll(z, -1, axis=0) * beta1
            elif mod == 1:
                inner_x_abs = np.abs(inner_x)
                temp = scipy.special.i1(inner_x_abs * 2 * N * beta1) / scipy.special.i0(inner_x_abs * 2 * N * beta1) * inner_x / inner_x_abs
                F1 += np.roll(temp, 1, axis=0)[:, :, None] * np.roll(z, 1, axis=0) * np.abs(beta1)
                F1 += np.conj(temp)[:, :, None] * np.roll(z, -1, axis=0) * np.abs(beta1)

        if self.Ly > 1:
            F += np.roll(U[:, :, 1], 1, axis=1)[:, :, None] * np.roll(z, 1, axis=1)
            F += np.conj(U[:, :, 1])[:, :, None] * np.roll(z, -1, axis=1)
            if mod == 0:
                F1 += np.roll(inner_y, 1, axis=1)[:, :, None] * np.roll(z, 1, axis=1) * beta1
                F1 += np.conj(inner_y)[:, :, None] * np.roll(z, -1, axis=1) * beta1
            elif mod == 1:
                inner_y_abs = np.abs(inner_y)
                temp = scipy.special.i1(inner_y_abs * 2 * N * beta1) / scipy.special.i0(inner_y_abs * 2 * N * beta1) * inner_y / inner_y_abs
                F1 += np.roll(temp, 1, axis=1)[:, :, None] * np.roll(z, 1, axis=1) * np.abs(beta1)
                F1 += np.conj(temp)[:, :, None] * np.roll(z, -1, axis=1) * np.abs(beta1)

        return N * beta * F * 2.0 + N * F1 * 2.0

    def _a_force(self):
        """
        Returns force for momentum equation da/dt = p_a/mass_a,
        dp_a/dt = force_a = -dS/da.
        """
        z = self.z
        a = self.a
        s = self.s
        alpha = self.alpha
        alpha1 = self.alpha1
        Ux = self.U[:, :, 0]
        Uy = self.U[:, :, 1]

        F = np.zeros((self.Lx, self.Ly, 2), dtype=complex)

        F[:, :, 0] += 2.0 * self.N * self.beta * np.imag(np.conj(Ux) * self._spin_inner_x())
        F[:, :, 1] += 2.0 * self.N * self.beta * np.imag(np.conj(Uy) * self._spin_inner_y())

        # ---------- Villain plaquette contribution ----------
        f = (a[:, :, 0]
            + np.roll(a, -1, axis=0)[:, :, 1]
            - np.roll(a, -1, axis=1)[:, :, 0]
            - a[:, :, 1]
            + 2 * np.pi * s)
        sinf = np.sin(f)

        # derivatives
        F[:, :, 0] -= (alpha * (f - np.roll(f, 1, axis=1)) + alpha1 * (sinf - np.roll(sinf, 1, axis=1)))
        F[:, :, 1] -= (alpha * (np.roll(f, 1, axis=0) - f) + alpha1 * (np.roll(sinf, 1, axis=0) - sinf))

        return F.real

    def _sample_momenta(self):
        p_a = np.sqrt(self.mass_a) * np.random.randn(self.Lx, self.Ly, 2)

        p_z = np.sqrt(self.mass_z) * (
            np.random.randn(self.Lx, self.Ly, self.N)
            + 1j * np.random.randn(self.Lx, self.Ly, self.N)
        )
        p_z = self._project_tangent(self.z, p_z)
        return p_a, p_z

    def _kinetic(self, p_a, p_z):
        K_a = 0.5 * np.sum(p_a * p_a) / self.mass_a
        K_z = 0.5 * np.sum(np.abs(p_z) ** 2) / self.mass_z
        return K_a + K_z

    def hamiltonian(self, p_a, p_z, mod=0):
        return self.action(mod=mod) + self._kinetic(p_a, p_z)

    def hmc_step(self, epsilon=None, n_leapfrog=None, mod=0):
        eps = self.epsilon if epsilon is None else float(epsilon)
        nlf = self.n_leapfrog if n_leapfrog is None else int(n_leapfrog)
        if eps <= 0:
            raise ValueError("epsilon must be positive")
        if nlf <= 0:
            raise ValueError("n_leapfrog must be positive")

        z_old = self.z.copy()
        a_old = self.a.copy()
        U_old = self.U.copy()
        s_old = self.s.copy()

        p_a, p_z = self._sample_momenta()
        H_old = self.hamiltonian(p_a, p_z, mod=mod)

        force_a = self._a_force()
        force_z = self._project_tangent(self.z, self._z_force(mod=mod))
        p_a += 0.5 * eps * force_a
        p_z += 0.5 * eps * force_z
        p_z = self._project_tangent(self.z, p_z)

        for step in range(nlf):
            # update a and z
            self.a += eps * p_a / self.mass_a
            self._sync_U_from_a()

            # rot_mat = scipy.linalg.expm(eps * (vouter_z(p_z, self.z) - vouter_z(self.z, p_z)) / self.mass_z) * np.exp(eps * vdot_z(p_z, self.z) / self.mass_z)[:, :, None, None]
            # self.z = np.einsum("ijkl,ijl->ijk", rot_mat, self.z)
            # p_z = np.einsum("ijkl,ijl->ijk", rot_mat, p_z)
            p_z_norm = np.linalg.norm(p_z, axis=-1)
            p_z_unit = p_z / p_z_norm[:, :, None]
            cos_p = np.cos(eps * p_z_norm / self.mass_z)
            sin_p = np.sin(eps * p_z_norm / self.mass_z)
            p_z = cos_p[:,:,None] * p_z - sin_p[:,:,None] * self.z * p_z_norm[:, :, None]
            self.z = cos_p[:,:,None] * self.z + sin_p[:,:,None] * p_z_unit
            self._normalize_z_inplace()
            p_z = self._project_tangent(self.z, p_z)

            # update momenta
            force_a = self._a_force()
            force_z = self._project_tangent(self.z, self._z_force(mod=mod))
            coeff = 1.0 if step < nlf - 1 else 0.5
            p_a += coeff * eps * force_a
            p_z += coeff * eps * force_z
            p_z = self._project_tangent(self.z, p_z)

        # deal with U(1) nature of a (wrap a into [-pi, pi] and update s so that
        # the physical plaquette variable da + 2*pi*s is invariant)
        self._apply_periodicity_wrap()

        H_new = self.hamiltonian(p_a, p_z, mod=mod)
        dH = H_new - H_old

        self.attempted_hmc += 1
        if np.log(np.random.rand()) < -dH:
            self.accepted_hmc += 1
            return True, dH

        self.z = z_old
        self.a = a_old
        self.U = U_old
        self.s = s_old
        return False, dH

    def _apply_periodicity_wrap(self):
        """Wrap a into [-pi, pi] and shift the integer field s on every plaquette
        so that the physical plaquette variable da + 2*pi*s is unchanged.

        PBC version: one plaquette per site, neighbours wrap periodically.
        """
        delta_a = np.rint((self.a - U1_clip(self.a)) / (2 * np.pi)).astype(int)
        self.a = U1_clip(self.a)
        self.s += (delta_a[:, :, 0]
                   + np.roll(delta_a, -1, axis=0)[:, :, 1]
                   - np.roll(delta_a, -1, axis=1)[:, :, 0]
                   - delta_a[:, :, 1])

    def metro_step(self, x, y, step):
        s_old = self.s[x, y]
        s_new = s_old + (int(np.abs(step * np.random.randn())) + 1) * np.random.choice((-1, 1))
        self.attempted_metro += 1
        # Hard-wall constraint |s| <= s_max: an out-of-range proposal is rejected
        # outright. The proposal stays symmetric, so detailed balance holds for the
        # truncated target pi(s) ~ exp(-S(s)) * 1[|s| <= s_max]. (Resampling the
        # proposal until it lands in range would make the proposal asymmetric near
        # +/-s_max and break detailed balance, since no Hastings correction is applied.)
        if np.abs(s_new) > self.s_max:
            return False, np.inf
        S_old = self.action_s(s_old, x, y)
        S_new = self.action_s(s_new, x, y)
        dS = S_new - S_old
        if np.random.rand() < np.exp(-dS):
            self.s[x, y] = s_new
            self.accepted_metro += 1
            return True, dS
        else:
            return False, dS

    def sweep(self, epsilon=None, n_leapfrog=None, s_step=None, s_update_num=None, mod=0):
        """Run one global HMC trajectory and several metropolis steps of s."""
        ss = self.s_step if s_step is None else float(s_step)
        sun = self.s_update_num if s_update_num is None else int(s_update_num)
        if ss <= 0:
            raise ValueError("s_step must be positive")
        if sun <= 0:
            raise ValueError("s_update_num must be positive")
        acc, dH = self.hmc_step(epsilon=epsilon, n_leapfrog=n_leapfrog, mod=mod)

        if np.abs(self.alpha) > 1e-8:
            indices = [(x, y, mu) for mu in range(sun) for x in range(self.Lx) for y in range(self.Ly)]
            np.random.shuffle(indices)
            for x, y, mu in indices:
                self.metro_step(x, y, step=ss)

        return acc, dH

    # ----------------------------------- Observables -----------------------------------
    
    def energy_density(self):
        """
        Compute S/V/2, S = sum (1 - |z' z|^2)

        Returns float.
        """
        z = self.z
        E1 = 1 - np.abs(vdot_z(np.roll(z, -1, 0), z))**2 / 2 - np.abs(vdot_z(np.roll(z, -1, 1), z))**2 / 2

        return np.mean(E1)

    def wilson_loop11(self):
        """
        Compute average value of a 1*1 Wilson loop i.e. Re(UUUU) = cos(da).

        Use two methods: Re(UUUU) and cos(arg(ZZ) + arg(ZZ) + ...), and return two float.
        """
        U = self.U
        z = self.z
        w_U = np.real(
            U[:, :, 0]
            * np.roll(U, -1, axis=0)[:, :, 1]
            * np.conj(np.roll(U, -1, axis=1)[:, :, 0] * U[:, :, 1])
        )
        temp = (
            vdot_z(z, np.roll(z, -1, 0))
            * vdot_z(np.roll(z, -1, 0), np.roll(z, -1, (0, 1)))
            * vdot_z(np.roll(z, -1, (0, 1)), np.roll(z, -1, 1))
            * vdot_z(np.roll(z, -1, 1), z)
        )
        w_z = np.cos(np.angle(temp))

        return 1 - np.mean(w_U), 1 - np.mean(w_z)

    def PP_corr_k(self):
        """
        Return fourier transformation of connected PP correlation.
        """
        N = self.N
        P = np.einsum("ijk,ijl->ijkl", self.z, np.conj(self.z))
        Px = P - 1 / N * np.eye(N)[None, None, :, :]
        Pk = fftn(Px, axes=(0, 1))
        S = np.sum(np.abs(Pk) ** 2, axis=(2, 3)) / self.V
        return S

    def conn_PP_corr(self):
        """Return real-space connected PP correlation, shape (Lx, Ly)."""
        return ifftn(self.PP_corr_k()).real

    def topo_charge(self):
        '''
        The skyrmion number of the system.

        Use three methods: arg(UUUU), (arg(ZZ) + arg(ZZ) + ...) and sum(s), and return (3,) array.
        '''
        U = self.U
        z = self.z

        Q_U = np.angle(
            U[:, :, 0]
            * np.roll(U, -1, axis=0)[:, :, 1]
            * np.conj(np.roll(U, -1, axis=1)[:, :, 0] * U[:, :, 1])
        )
        temp0 = (
            vdot_z(z, np.roll(z, -1, 0))
            * vdot_z(np.roll(z, -1, 0), np.roll(z, -1, (0, 1)))
            * vdot_z(np.roll(z, -1, (0, 1)), z)
        )
        temp1 = (
            vdot_z(z, np.roll(z, -1, (0, 1)))
            * vdot_z(np.roll(z, -1, (0, 1)), np.roll(z, -1, 1))
            * vdot_z(np.roll(z, -1, 1), z)
        )
        Q_z = np.angle(temp0) + np.angle(temp1)

        Q = np.asarray([int(np.round(np.sum(Q_U) / 2 / np.pi)), int(np.round(np.sum(Q_z) / 2 / np.pi)), np.sum(self.s)])

        return Q

if __name__ == "__main__":
    Lx, Ly, N = 40, 40, 4
    beta, beta1, alpha, alpha1 = 1.0, 0.0, 0.0, 0.0
    sampler = CPN_RefVil_HMCSampler(N, Lx, Ly, beta, beta1, alpha, alpha1, seed=None)
    eps = 0.02
    nlf = 50
    for _ in tqdm(range(100)):
        sampler.sweep(epsilon=eps, n_leapfrog=nlf, s_step=0.5, s_update_num=10, mod=0)
    # print(sampler.topo_density_corr())
    # print(sampler.topo_density_corr_fft())
    print(f"Acceptance Rate: {sampler.accept_rate}")
