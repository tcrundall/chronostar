import numpy as np

class SynthGroup():
    def SynthGroup(self, pars, sphere=True):
        # If sphere flag is set, interpret pars one way
        # If not set, interpret pars another way
        # Simply supposed to be a neat way of packaging up a group's initial
        # conditions
        self.is_sphere = sphere
        if sphere:
            self.mean = pars[:6]
            self.dx = self.sphere_dx = pars[6]
            self.dv = pars[7]
            self.age = pars[8]
            self.nstars = pars[9]
        else:
            self.mean = pars[:6]
            self.dx, self.dy, self.dz = pars[6:9]
            self.dv = pars[9]
            self.cxy, self.cxz, self.cyz = pars[10:13]
            self.age = pars[13]
            self.nstars = pars[14]

            self.sphere_dx = (self.dx * self.dy * self.dz)**(1./3.)

    def getSphericalPars(self):
        return np.hstack((self.mean, self.sphere_dx, self.dv, self.age))


    def getFreePars(self):
        if self.is_sphere:
            return np.hstack((self.mean, self.dx, self.dx, self.dx, self.dv,
                             0.0, 0.0, 0.0, self.age))
        else:
            return np.hstack((self.mean, self.dx, self.dy, self.dz, self.dv,
                             self.cxy, self.cxz, self.cyz, self.age))

    def generateSphericalCovMatrix(self):
        dx = self.sphere_dx
        dv = self.dv
        scmat = np.array([
            [dx**2, 0., 0., 0., 0., 0.],
            [0., dx**2, 0., 0., 0., 0.],
            [0., 0., dx**2, 0., 0., 0.],
            [0., 0., 0., dv**2, 0., 0.],
            [0., 0., 0., 0., dv**2, 0.],
            [0., 0., 0., 0., 0., dv**2],
        ])
        return scmat

    def generateEllipticalCovMatrix(self):
        if self.is_sphere:
            return self.generateSphericalCovMatrix()
        else:
            dx, dy, dz = self.dx, self.dy, self.dz
            dv = self.dv
            cxy, cxz, cyz = self.cxy, self.cxz, self.cyz
            ecmat = np.array([
                [dx**2, cxy*dx*dy, cxz*dx*dz, 0., 0., 0.],
                [cxy*dx*dy, dy**2, cyz*dy*dz, 0., 0., 0.],
                [cxz*dx*dz, cyz*dy*dz, dz**2, 0., 0., 0.],
                [       0.,        0.,    0., dv**2, 0., 0.],
                [       0.,        0.,    0., 0., dv**2, 0.],
                [       0.,        0.,    0., 0., 0., dv**2],
            ])
            assert np.allclose(ecmat, ecmat.T)
            return ecmat



