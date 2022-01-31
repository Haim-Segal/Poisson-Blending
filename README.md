# Poisson Blending
## Blending images using Poisson’s equation and sparse matrices
Poisson Blending is an algorithm used in image processing for achieving a realistic looking composition of two images using Discrete Poisson equation.
The key idea is to approach this challenge as Dirichlet problem. In particular, let us treat the overlapping region between the images (a source image being inserted into a destination image) as the potential field caused by a given density distribution of electric charge or mass which in our case will be related to the laplacian of the images, and as the fixed boundary condition we will take the values of the destination image on the boundary curve enclosing the source image patch.

Finally, we deal with a linear system of equations represented by Ax=b, where the size of the coefficient matrix A is NxN, where N is the number of the pixels in the source image.
In order to construct and solve this system I use the great library of Algebraic Multigrid (AMG) solvers [PyAMG](https://github.com/pyamg/pyamg).

## Results
naive blending
![Naive Blended](https://user-images.githubusercontent.com/82455000/150935595-3787de6a-f4e0-44dc-87e5-3c57b72ab71c.png)

Poisson blending
![Poisson Blended](https://user-images.githubusercontent.com/82455000/150935633-ce21a68e-1910-461b-9f24-871f09e03ea4.png)
 
original images links: 
[source image](https://www.pentaxforums.com/gallery/photo-old-airplane-3-12687/&u=15401), 
[destination image](https://www.azocleantech.com/news.aspx?newsID=26805)

## Installation and Usage
run Poisson Blending 2D.py and follow the instructions

Enjoy!
## References
[Patrick Perez, Michel Gangnet, Andrew Blake, "Poisson Image Editing".](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf)
