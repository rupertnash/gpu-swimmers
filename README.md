# Subgrid swimmers with lattice Boltzmann, on a GPU

This software is copyright Rupert Nash and provided under the terms specified in LICENSE.txt

This is based on https://github.com/rupertnash/subgrid

Includes:
* LB solver (D3Q15, MRT)
* Arrays of point-like self-propelled particles
* Arrays of point-like tracer particles (that do not affect the fluid)

#References:
* Nash, R. W. (2010, January 28). Efficient lattice Boltzmann simulations of self-propelled particles with singular forces. University of Edinburgh. http://hdl.handle.net/1842/4143
* Nash, R. W., Adhikari, R., & Cates, M. E. (2008). Singular forces and pointlike colloids in lattice Boltzmann hydrodynamics. Physical Review E, 77(2), 026709. [doi:10.1103/PhysRevE.77.026709](http://dx.doi.org/10.1103/PhysRevE.77.026709)
* Nash, R. W., Adhikari, R., Tailleur, J., & Cates, M. E. (2010). Run-and-tumble particles with hydrodynamics: Sedimentation, trapping, and upstream swimming. Physical Review Letters, 104(25), 258101. [doi:10.1103/PhysRevLett.104.258101](http://dx.doi.org/10.1103/PhysRevLett.104.258101)
