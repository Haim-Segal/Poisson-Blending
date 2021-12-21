# Haim Segal
import numpy as np
from scipy.sparse import csr_matrix
import PIL
from matplotlib import pyplot as plt
from pyamg.gallery import poisson
import pyamg


def readImageToRgb(image):
    return PIL.Image.Image.split(PIL.Image.open(image))


def readImages(srcImgName, dstImgName):
    srcColorImg = PIL.Image.open(srcImgName)
    srcRgbImg = PIL.Image.Image.split(srcColorImg)
    dstColorImg = PIL.Image.open(dstImgName)
    dstRgbImg = PIL.Image.Image.split(dstColorImg)
    minColor = 0
    maxColor = 255
    return srcRgbImg, dstRgbImg, minColor, maxColor

def poisson_blending(srcImgName, dstImgName, mixedGrad=True, linearCombination=False, product=0.5, weight=0.5):

    def locateSrcInDst():
        # center = np.asarray(dstSize) // 2
        center = (dstSize[0] // 2 - 200, dstSize[1] // 2 - 350)
        corner_ = (center[0] - srcSize[0] // 2, center[1] - srcSize[1] // 2)
        relevant__dst = dstImg[
                       corner_[0]:corner_[0] + srcSize[0],
                       corner_[1]:corner_[1] + srcSize[1]]
        return relevant__dst, corner_

    def buildLinearSystem():

        def laplacian(array):
            return (poisson(array.shape, format='csr') * csr_matrix(array.flatten()).transpose()).toarray()

        def construct_const_vector():
            if mixedGrad:
                dst_laplacianed = laplacian(relevantDst)
                if linearCombination:
                    b__ = np.reshape(
                        weight * src_laplacianed + (1 - weight) * dst_laplacianed, srcSize)
                else:
                    bigger_laplacian = abs(src_laplacianed) >= product * abs(dst_laplacianed)
                    b__ = np.reshape(bigger_laplacian * src_laplacianed +
                                     (1 - weight) * ~bigger_laplacian * dst_laplacianed +
                                     weight * ~bigger_laplacian * src_laplacianed, srcSize)
            else:
                b__ = np.reshape(src_laplacianed, srcSize)
            b__[1, :] = relevantDst[1, :]
            b__[-2, :] = relevantDst[-2, :]
            b__[:, 1] = relevantDst[:, 1]
            b__[:, -2] = relevantDst[:, -2]
            b__ = b__[1:-1, 1: -1]
            return b__, b__.shape

        def construct_coefficient_mat(size):

            def fix_boundary_condition(coeff):
                size_prod = np.prod(np.asarray(size))
                arange_space = np.arange(size_prod).reshape(size)
                arange_space[1:-1, 1:-1] = -1
                index_to_change = arange_space[arange_space > -1]
                for j in index_to_change:
                    coeff[j, j] = 1
                    if j - 1 > -1:
                        coeff[j, j - 1] = 0
                    if j + 1 < size_prod:
                        coeff[j, j + 1] = 0
                    if j - size[-1] > - 1:
                        coeff[j, j - size[-1]] = 0
                    if j + size[-1] < size_prod:
                        coeff[j, j + size[-1]] = 0
                return coeff

            a__ = poisson(size, format='lil')
            a__ = fix_boundary_condition(a__)
            return a__

        src_laplacianed = laplacian(srcImg)
        b_, b__size = construct_const_vector()
        a_ = construct_coefficient_mat(b__size)
        return a_, b_, b__size

    def solveLinearSystem():
        multi_level = pyamg.ruge_stuben_solver(csr_matrix(a))
        x_ = multi_level.solve(b, tol=1e-10).reshape(b_size)
        x_[x_ < minColor] = minColor
        x_[x_ > maxColor] = maxColor
        return x_

    def constructMixedAndNaive():
        mixed = dstImg.copy()
        naive = dstImg.copy()
        mixed[corner[0] + 1:corner[0] + 1 + b_size[0], corner[1] + 1:corner[1] + 1 + b_size[1]] = x
        naive[corner[0]:corner[0] + srcSize[0], corner[1]:corner[1] + srcSize[1]] = srcImg
        poissonBlended.append(PIL.Image.fromarray(mixed))
        naive_blended.append(PIL.Image.fromarray(naive))

    srcRgbImg, dstRgbImg, minColor, maxColor = readImages(srcImgName, dstImgName)
    poissonBlended = []
    naive_blended = []
    for color in range(3):
        srcImg = np.asarray(srcRgbImg[color])
        dstImg = np.asarray(dstRgbImg[color])
        srcSize = srcImg.shape
        dstSize = dstImg.shape
        relevantDst, corner = locateSrcInDst()
        a, b, b_size = buildLinearSystem()
        x = solveLinearSystem()
        constructMixedAndNaive()
    poissonBlended = PIL.Image.merge('RGB', tuple(poissonBlended))
    poissonBlended.save('poissonBlended.png')
    naive_blended = PIL.Image.merge('RGB', tuple(naive_blended))
    naive_blended.save('naive_blended.png')
    poissonBlended.show('poissonBlended')
    naive_blended.show('naive_blended')


def main():
    poisson_blending('src_img/old_airplane3.jpg', 'dst_img/underwater5.jpg', True, False, 1, 0)

if __name__ == '__main__':
    main()
