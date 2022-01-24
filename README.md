# Poisson Blending
## Blending images using Poisson’s equation and sparse matrices
Poisson Blending is an algorithm used in image processing for achieving a realistic looking composition of two images using Discrete Poisson equation.
The key idea is to approach this challenge as Dirichlet problem. In particular, let us treat the overlapping region between the images (a source image being inserted into a destination image) as the potential field caused by a given density distribution of electric charge or mass which in our case will be related to the laplacian of the images, and as the fixed boundary condition we will take the values of the destination image on the boundary curve enclosing the source image patch.
Finally, we deal with a linear system of equations represented by Ax=b, where the size of the coefficient matrix A is NxN, where N is the number of the pixels in the source image. In order to handle the large storage requirements needed to solve this system we use Sparse Matrices.
## References
[Patrick Perez, Michel Gangnet, Andrew Blake, "Poisson Image Editing".](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf)
