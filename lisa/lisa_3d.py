# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

import sys
import pickle
import numpy as np
import matplotlib.cm as cm
from numpy import linalg as la
import matplotlib.pyplot as plt
from scipy import integrate as sint


# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

class NMA_3D:

    def __init__(
        self,
        ELN_params: dict,
        mu=1.0
    ) -> None:

        self.mu = mu
        self.ELN = ELN_params["ELN"]

        assert 'vz_range' in ELN_params.keys(), "Require key 'vz_range' in ELN_params."
        assert isinstance(ELN_params['vz_range'], (list, tuple, np.ndarray)
                          ), 'Type of ELN_params["vz_range"] must be from (list, tuple, np.ndarray)'
        assert 'phi_range' in ELN_params.keys(
        ), "Require key 'phi_range' in phasespace_range."
        assert isinstance(ELN_params['phi_range'], (list, tuple, np.ndarray)
                          ), 'type(ELN_params["phi_range"]) is not in (list, tuple, np.ndarray)'
        assert 'nvz' in ELN_params.keys(), 'Require key "nvz" in phasespace_grids'
        assert 'nphi' in ELN_params.keys(), 'Require key "nphi" in ELN_params'
        assert isinstance(ELN_params['nvz'],
                          int), 'ELN_params[nvz] must be an integer'
        assert isinstance(ELN_params['nphi'],
                          int), 'ELN_params[nph] must be an integer'

        self.vz_min, self.vz_max = ELN_params['vz_range']
        self.phi_min, self.phi_max = ELN_params['phi_range']

        assert self.vz_max > self.vz_min, 'vz_max must be larger than vz_min'
        assert self.phi_max > self.phi_min, 'phi_max must be larger than phi_min'

        self.nvz, self.nphi = ELN_params['nvz'], ELN_params['nphi']
        self.dvz, self.dphi = (self.vz_max-self.vz_min) / \
            self.nvz, (self.phi_max-self.phi_min)/self.nphi
        self.vz = self.vz_min + (np.arange(0, self.nvz) + 0.5)*self.dvz
        self.phi = self.phi_min + (np.arange(0, self.nphi) + 0.5)*self.dphi

        return None

    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

    def eps_0(self):  # [Tested; stat:Success]
        def e_0(v, phi): return 1*self.ELN(v, phi)
        return sint.dblquad(
            e_0,
            self.phi_min, self.phi_max,
            self.vz_min, self.vz_max
        )[0]*(1.0/(2.0*np.pi))

    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

    def eps_x(self):  # [Tested; stat:Success]
        return sint.dblquad(
            lambda v, phi: np.sqrt(1-v**2)*np.cos(phi)*self.ELN(v, phi),
            self.phi_min, self.phi_max,
            self.vz_min, self.vz_max
        )[0]*(1.0/(2.0*np.pi))

    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

    def eps_y(self):  # [Tested; stat:Success]
        return sint.dblquad(
            lambda v, phi: np.sqrt(1-v**2)*np.sin(phi)*self.ELN(v, phi),
            self.phi_min, self.phi_max,
            self.vz_min, self.vz_max
        )[0]*(1.0/(2.0*np.pi))

    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

    def eps_z(self):  # [Tested; stat:Success]
        return sint.dblquad(
            lambda v, phi: v*self.ELN(v, phi),
            self.phi_min, self.phi_max,
            self.vz_min, self.vz_max
        )[0]*(1.0/(2.0*np.pi))

    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

    def coeff_matrices_sym_pres(self):

        A0 = np.identity(self.nvz*self.nphi)

        Ax = np.diag(
            np.array(
                [np.sqrt(1-self.vz[int(i/self.nphi)]**2)*np.cos(self.phi[int(i % self.nphi)])
                 for i in range(self.nvz*self.nphi)]
            )
        )

        Ay = np.diag(
            np.array(
                [np.sqrt(1-self.vz[int(i/self.nphi)]**2)*np.sin(self.phi[int(i % self.nphi)])
                 for i in range(self.nvz*self.nphi)]
            )
        )

        Az = np.diag(
            np.array(
                [self.vz[int(i/self.nphi)]
                 for i in range(self.nphi*self.nvz)]
            )
        )

        I0 = np.outer(
            np.ones(self.nphi*self.nvz),
            np.array(
                [self.ELN(self.vz[int(i/self.nphi)], self.phi[int(i % self.nphi)])
                 for i in range(self.nphi*self.nvz)]
            ).reshape(1, -1)
        )

        Ix = np.outer(
            np.array(
                [np.sqrt(1 - self.vz[int(i/self.nphi)]**2)*np.cos(self.phi[int(i % self.nphi)])
                 for i in range(self.nphi*self.nvz)]
            ).reshape(-1, 1),
            np.array(
                [np.sqrt(1 - self.vz[int(i/self.nphi)]**2)*np.cos(self.phi[int(i % self.nphi)]) * self.ELN(
                    self.vz[int(i/self.nphi)], self.phi[int(i % self.nphi)]) for i in range(self.nvz*self.nphi)]
            ).reshape(1, -1)
        )

        Iy = np.outer(
            np.array(
                [np.sqrt(1 - self.vz[int(i/self.nphi)]**2)*np.sin(self.phi[int(i % self.nphi)])
                 for i in range(self.nphi*self.nvz)]
            ).reshape(-1, 1),
            np.array(
                [np.sqrt(1 - self.vz[int(i/self.nphi)]**2)*np.sin(self.phi[int(i % self.nphi)]) * self.ELN(
                    self.vz[int(i/self.nphi)], self.phi[int(i % self.nphi)]) for i in range(self.nvz*self.nphi)]
            ).reshape(1, -1)
        )

        Iz = np.outer(
            np.array(
                [self.vz[int(i/self.nphi)]
                 for i in range(self.nvz*self.nphi)]
            ).reshape(-1, 1),
            np.array(
                [self.vz[int(i/self.nphi)] * self.ELN(self.vz[int(i/self.nphi)],
                                                      self.phi[int(i % self.nphi)]) for i in range(self.nvz*self.nphi)]
            ).reshape(1, -1)
        )

        print(f"Max I0: {np.abs(I0).max():.4E}")
        print(f"Max Ix: {np.abs(Ix).max():.4E}")
        print(f"Max Iy: {np.abs(Iy).max():.4E}")
        print(f"Max Iz: {np.abs(Iz).max():.4E}")

        return (A0, Ax, Ay, Az, I0, Ix, Iy, Iz)

    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

    def coeff_matrices_sym_break(self):

        A0 = np.identity(self.nvz*self.nphi)

        Ax = np.diag(
            np.array(
                [np.sqrt(1-self.vz[int(i/self.nphi)]**2)*np.cos(self.phi[int(i % self.nphi)])
                 for i in range(self.nvz*self.nphi)]
            )
        )

        Az = np.diag(
            np.array(
                [self.vz[int(i/self.nphi)]
                 for i in range(self.nphi*self.nvz)]
            )
        )

        Iy = np.outer(
            np.array(
                [np.sqrt(1 - self.vz[int(i/self.nphi)]**2)*np.sin(self.phi[int(i % self.nphi)])
                 for i in range(self.nphi*self.nvz)]
            ).reshape(-1, 1),
            np.array(
                [np.sqrt(1 - self.vz[int(i/self.nphi)]**2)*np.sin(self.phi[int(i % self.nphi)]) * self.ELN(
                    self.vz[int(i/self.nphi)], self.phi[int(i % self.nphi)]) for i in range(self.nvz*self.nphi)]
            ).reshape(1, -1)
        )

        print(f"Max Iy: {np.abs(Iy).max():.4E}")

        return (A0, Ax, Az, 0, 0, Iy, 0)

    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

    def characteristic_matrix(self, matrices, epsilons,  kx=0, ky=0, kz=0):

        e0, ex, ey, ez = epsilons
        A0, Ax, Ay, Az, I0, Ix, Iy, Iz = matrices

        return (
            e0*A0 + (kx-ex)*Ax + (ky-ey)*Ay + (kz-ez)*Az
            - self.mu*(self.dvz * self.dphi/(2*np.pi))*(I0 - Ix - Iy - Iz)
        )

    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

    def evaluate_normal_modes(
        self,
        kxs: np.ndarray,
        kys: np.ndarray,
        kzs: np.ndarray,
        charMs: np.ndarray
    ) -> dict:

        e0 = self.eps_0()
        ex = self.eps_x()
        ey = self.eps_y()
        ez = self.eps_z()

        print(
            f"e0: {e0:.4E}\n" +
            f"ex: {ex:.4E}\n" +
            f"ey: {ey:.4E}\n" +
            f"ez: {ez:.4E}\n"
        )

        epsilons = (e0, ex, ey, ez)
        o_r = np.zeros((len(kxs), len(kys), len(kzs)))
        o_i = np.zeros((len(kxs), len(kys), len(kzs)))

        with open("analysis.dat", "w") as f:
            sys.stdout.flush()
            # progress = ['/', '-', '\\']
            progress = [
                "[=     ]",
                "[ =    ]",
                "[  =   ]",
                "[   =  ]",
                "[    = ]",
                "[     =]",
                "[    = ]",
                "[   =  ]",
                "[  =   ]",
                "[ =    ]",
            ]

            l = len(kxs) * len(kxs) * len(kxs)
            for ikz, kz in enumerate(kzs):
                for iky, ky in enumerate(kys):
                    for ikx, kx in enumerate(kxs):
                        current_itr = ikz * iky * ikx

                        if ikx % 10 == 0:
                            print(
                                f"\r{progress[ikz%len(progress)]} {int(ikz*100/len(kzs))}% [{ikz=} {iky=} {ikx=}]", end="")
                            sys.stdout.flush()

                        M = self.characteristic_matrix(
                            charMs, epsilons, kx=kx, ky=ky, kz=kz)

                        eival = la.eig(M)[0]
                        evi = 0.0
                        evr = 0.0
                        evi = np.abs(eival.imag).max()
                        # Index of the max growth rate
                        idx = np.argmax(np.abs(eival.imag))
                        evr = eival.real.max()

                        o_i[ikx][iky][ikz] = np.abs(evi)
                        o_r[ikx][iky][ikx] = np.abs(evr)

            print(f"\r{'['}{'='*3}{'='*3}] {100}%", end="")
            sys.stdout.flush()

            result = {
                'O_i': o_i,
                'O_r': o_r,
                'kx': kxs,
                'ky': kys,
                'kz': kzs
            }
        return result

    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

    def write_dict(self, dct: dict, outpath: str) -> None:
        with open(outpath, 'wb') as ofstream:
            pickle.dump(dct, ofstream, pickle.HIGHEST_PROTOCOL)
        return None

    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

    def plot_eln(self, store_to) -> None:
        eln_arr = np.zeros((self.nvz, self.nphi))
        for vzid, vz in enumerate(self.vz):
            for phid, phi in enumerate(self.phi):
                eln_arr[vzid][phid] = self.ELN(vz, phi)

        plt.contourf(self.phi/np.pi, self.vz, eln_arr,
                     levels=self.nvz, cmap="RdBu_r")
        plt.colorbar()
        plt.savefig(store_to, dpi=150)
        print(f"ELN plot saved to {store_to}")
        return

    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

    def run(self, kxs, kys, kzs, store_to, mode='sym') -> None:

        self.plot_eln(f"ELN_{''.join(store_to.split('.')[:-1])}.png")
        if mode == 'sym':
            charMs = self.coeff_matrices_sym_pres()

        if mode == 'break':
            charMs = self.coeff_matrices_sym_break()

        result_sym = self.evaluate_normal_modes(kxs, kys, kzs, charMs)
        self.write_dict(result_sym, store_to)
        return None

    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
