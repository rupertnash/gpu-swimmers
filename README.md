# Subgrid swimmers with lattice Boltzmann, on a GPU

This software is copyright Rupert Nash and provided under the terms specified in LICENSE.txt

This is based on https://github.com/rupertnash/subgrid

## Includes:
* LB solver (D3Q15, MRT)
* Arrays of point-like self-propelled particles
* Arrays of point-like tracer particles (that do not affect the fluid)

## Requirements:
* CMake
* Python
* NumPy
* Cython
* TargetDP https://ccpforge.cse.rl.ac.uk/gf/project/ludwig/scmsvn/?action=browse&path=%2Ftrunk%2FtargetDP%2F
* CUDA to use a GPU
* OpenMP to use multiple cores

## Building
Done with CMake
```
git clone https://github.com/rupertnash/gpu-swimmers.git
mkdir gpu-swimmers-build
cd gpu-swimmers-build
cmake -DLBSWIMMERS_PACKAGE_INSTALL_DIR=somewhere/on/PYTHONPATH ../gpu-swimmers
make install
```

## Structure
The core LB and swimmers are implemented in C++ in lbswim/liblbswim. This is compiled to a shared library `liblbswim.so`. All the classes have both CPU/host and GPU/device copies of their data. It is up to the user to update things, but the library makes this fairly simple by offering `D2H()` (device to host) and `H2D()` (host to device) methods to move data.

This is then wrapped by Cython and made available in Python as the package `lbswim`.

## Differences from subgrid
* Only periodic boundary conditions
* Only swimmers with tumbling and tracers
* Lattice sites run from (0, SIZE-1) not (1, SIZE)
* Particle positions run from [0., SIZE)
* All vector data is laid out with the element index being slowest varying. I.e. a three vector is laid out as [x0, x1, ..., xn-1, y0, y1, .. yn-1, z0, z1, zn-1]
* Must manually move data between host and device
* Pickling of objects is not implemented yet

## Usage
1. Create a Lattice of the desired size and relaxation parameters.
  1. Initialise the host's copy of the hydrodynamic fields (rho, u, force)
  2. Send them to the device
2. Create any swimmers needed.
  1. Create a CommonParams object and set the values
    * P - propelling force
    * l - swimmer length
    * alpha - tumble probabilty
    * mobility - swimmer self mobility - see below.
    * translational_advection_off - flag to turn off advection of swimmers by fluid
    * rotational_advection_off - flag to turn off rotation of swimmers by fluid
    * seed - seed for the random number generator
    The mobility is typically  (1 / a - 1/ hydro) / (6 pi eta), where a = nominal body size, hydro = correction and eta = fluid viscosity.
  2. Create the arrays with `SwimmerArray(nSwim, params)`
  3. Set the host copies of variables (noting that the data is laid out with shape (NDIMS, NSWIM):
    * r - position
    * n - orientation unit vector
    * v - velocity
  4. Send the data to the device
3. Create any tracers needed
  1. Create with `TracerArray(nPart)`
  2. Set host copies of r (position) and v (velocity)
  3. Send data to the device
4. Initialise the distributions with `lattice.InitFromHydro()`
5. Do some steps like (or see Python notes below).
  1. lat.ZeroForce()
  2. swimmers.AddForces(self.lat)
  3. lat.Step()
  4. lat.CalcHydro()
  5. swimmers.Move(lat)
  6. tracers.Move(lat)
6. Copy data you want to look at to the host

### C++
Sorry, no docs yet.

### Python
Very similar to above. Uses Numpy to wrap the data arrays. Uses properties to make data accessible. An example is probably best:
```
import numpy as np
from lbswim import lb, swimmers, tracers
from lbswim.System import System


tau = 0.25
eta = tau / 3
lat = lb.Lattice(16, 16, 16, tau, tau)
lat.rho[:] = 1.0
lat.rho.H2D()
lat.u[:] = 0.0
lat.u.H2D()
lat.ZeroForce()

system = System(lat)

cp = swimmers.CommonParams()
cp.P = 1e-3
cp.l = 0.1
cp.alpha = 0.01

a = 0.1
hydroRadius = 1.5
cp.mobility = (1.0 / a - 1.0 / hydroRadius) / (6.0 * np.pi * eta)
cp.translational_advection_off = False
cp.rotational_advection_off = False
cp.seed = 1234
sw = swimmers.Array(1, cp)
sw.r[:] = 8
sw.n[:] = [[0],[0],[1]]
sw.v[:] = 0.
sw.H2D()
system.AddThing(sw)

tr = tracers.Array(16**3)
tr.r[:] = np.mgrid[:16,:16,:16].reshape(3, 16**3)
tr.v[:] = 0.
tr.H2D()
system.AddThing(tr)

lat.InitFromHydro()

while lat.time_step < 1000:
    if lat.time_step % 10 == 0:
        sw.D2H()
        print sw.r[:,0]
    system.Step()
```


## References:
* Nash, R. W. (2010, January 28). Efficient lattice Boltzmann simulations of self-propelled particles with singular forces. University of Edinburgh. http://hdl.handle.net/1842/4143
* Nash, R. W., Adhikari, R., & Cates, M. E. (2008). Singular forces and pointlike colloids in lattice Boltzmann hydrodynamics. Physical Review E, 77(2), 026709. [doi:10.1103/PhysRevE.77.026709](http://dx.doi.org/10.1103/PhysRevE.77.026709)
* Nash, R. W., Adhikari, R., Tailleur, J., & Cates, M. E. (2010). Run-and-tumble particles with hydrodynamics: Sedimentation, trapping, and upstream swimming. Physical Review Letters, 104(25), 258101. [doi:10.1103/PhysRevLett.104.258101](http://dx.doi.org/10.1103/PhysRevLett.104.258101)
