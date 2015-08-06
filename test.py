import sys
import pdb

import numpy as np
import lb
import swimmers


lat = lb.Lattice(16,16,16, 0.25, 0.25)

lat.rho[:] = 1.0
lat.rho.H2D()

lat.u[:] = 0.0
lat.u.H2D()

lat.ZeroForce()

lat.InitFromHydro()

sw = swimmers.Array(1, 1.5)

sw.r[:] = 8
sw.n[:] = [[0],[0],[1]]
sw.v[:] = 0.
sw.P[:] = 1e-3
sw.a[:] = 0.1
sw.l[:] = 0.1
sw.H2D()

s = swimmers.System(lat, sw)

while s.lat.time_step < 100:
    s.Step()
    if s.lat.time_step % 10 == 0:
        s.sw.D2H()
        print s.sw.r[:,0]
