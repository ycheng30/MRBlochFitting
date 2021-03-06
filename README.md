# MRBlochFitting
## MR Data processing pipeline in Python
Some of the functions defined in the library involve numerical curve fitting, linear/nonlinear regression, fast fourier transform, read in high-dimensional large-scale image files, image processing including image registration and interpolation. <br />
<br />

> ifftprocessing.py

Includes functions that perform fast fourier transform <br />

<br />

> cjlib.py

Defines ~100 functions that perform reading high-dimensional large-scale image files with different format/headers, numerical curve fitting, optimization algorithms (gradient descent, BFGS, LBFGS, etc), convolutions using Gaussician kernels, image smoothing, image alignment/registration, phase unwrapping, data interpolation, extrapolation. Python packages involved are numpy, scipy, matplotlib. <br /> 

<br />

> bloch_fast.py and bloch_fast_Ying.py

Defines functions to perform Bloch equation simulation, which involve parameters initialization for different pool types (water, tissue, bound proteins), numerical methods and curve fitting. 
