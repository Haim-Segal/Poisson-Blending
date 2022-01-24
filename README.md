# Poisson Blending
## Blending images using Poisson’s equation and sparse matrices
Poisson Blending is an algorithm used in image processing for creating a realistic looking composition of two images using Discrete Poisson equation.
The key idea is to approach this challenge as Dirichlet problem. In particular, let us treat the overlapping region between the images (a source image being inserted into a destination image) as the potential field caused by a given electric charge or mass density distribution which in our case will be related to the laplacian of the images, and as the fixed boundary condition we will take the values of the destination image on the boundary curve enclosing the source image patch.

## References
[Patrick Perez, Michel Gangnet, Andrew Blake, "Poisson Image Editing".](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf)
