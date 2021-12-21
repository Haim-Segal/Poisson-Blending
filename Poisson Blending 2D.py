# Haim Segal
import numpy as np
from scipy.sparse import csr_matrix
import PIL
from matplotlib import pyplot
from pyamg.gallery import poisson
import pyamg


def imageToRgb(imagePath):
    return PIL.Image.Image.split(PIL.Image.open(imagePath))

def topLeftCornerOfSrcOnDst(srcImgShape, dstImgShape, horizontalBias=-200, verticalBias=-350):
    center = (dstImgShape[0] // 2 + horizontalBias, dstImgShape[1] // 2 + verticalBias)
    corner = (center[0] - srcImgShape[0] // 2, center[1] - srcImgShape[1] // 2)
    return corner

def cropDstUnderSrc(dstImg, corner, srcImgShape):
    dstUnderSrc = dstImg[
                   corner[0]:corner[0] + srcImgShape[0],
                   corner[1]:corner[1] + srcImgShape[1]]
    return dstUnderSrc

def laplacian(array):
    return (poisson(array.shape, format='csr') * csr_matrix(array.flatten()).transpose()).toarray()

def construct_const_vector(mixedGrad, linearCombination, dstUnderSrc, srcLaplacianed, weight, srcImgShape, product):
    if mixedGrad:
        dstLaplacianed = laplacian(dstUnderSrc)
        if linearCombination:
            b = np.reshape(
                weight * srcLaplacianed + (1 - weight) * dstLaplacianed, srcImgShape)
        else:
            biggerLaplacian = abs(srcLaplacianed) >= product * abs(dstLaplacianed)
            b = np.reshape(biggerLaplacian * srcLaplacianed +
                           (1 - weight) * ~biggerLaplacian * dstLaplacianed +
                           weight * ~biggerLaplacian * srcLaplacianed, srcImgShape)
    else:
        b = np.reshape(srcLaplacianed, srcImgShape)
    b[1, :] = dstUnderSrc[1, :]
    b[-2, :] = dstUnderSrc[-2, :]
    b[:, 1] = dstUnderSrc[:, 1]
    b[:, -2] = dstUnderSrc[:, -2]
    b = b[1:-1, 1: -1]
    return b, b.shape

def fixBoundaryCondition(coeff, shape):
    shapeProd = np.prod(np.asarray(shape))
    arangeSpace = np.arange(shapeProd).reshape(shape)
    arangeSpace[1:-1, 1:-1] = -1
    indexToChange = arangeSpace[arangeSpace > -1]
    for j in indexToChange:
        coeff[j, j] = 1
        if j - 1 > -1:
            coeff[j, j - 1] = 0
        if j + 1 < shapeProd:
            coeff[j, j + 1] = 0
        if j - shape[-1] > - 1:
            coeff[j, j - shape[-1]] = 0
        if j + shape[-1] < shapeProd:
            coeff[j, j + shape[-1]] = 0
    return coeff

def construct_coefficient_mat(shape):
    a = poisson(shape, format='lil')
    a = fixBoundaryCondition(a, shape)
    return a

def buildLinearSystem(srcImg, dstUnderSrc, mixedGrad, linearCombination, weight, product):
    srcLaplacianed = laplacian(srcImg)
    b, bShape = construct_const_vector(mixedGrad, linearCombination, dstUnderSrc, srcLaplacianed, weight,
                                       srcImg.shape, product)
    a = construct_coefficient_mat(bShape)
    return a, b, bShape

def solveLinearSystem(a, b, bShape, minColor, maxColor):
    multi_level = pyamg.ruge_stuben_solver(csr_matrix(a))
    x = multi_level.solve(b, tol=1e-10).reshape(bShape)
    x[x < minColor] = minColor
    x[x > maxColor] = maxColor
    return x

def constructMixedAndNaive(dstImg, srcImg, corner, x, bShape, poissonBlended, naiveBlended):
    mixed = dstImg.copy()
    naive = dstImg.copy()
    mixed[corner[0] + 1:corner[0] + 1 + bShape[0], corner[1] + 1:corner[1] + 1 + bShape[1]] = x
    naive[corner[0]:corner[0] + srcImg.shape[0], corner[1]:corner[1] + srcImg.shape[1]] = srcImg
    poissonBlended.append(PIL.Image.fromarray(mixed))
    naiveBlended.append(PIL.Image.fromarray(naive))

def mergeSaveShow(splittedImg, ImgName, ImgTitle):
    blended = PIL.Image.merge('RGB', tuple(splittedImg))
    blended.save(ImgName)
    blended.show(ImgTitle)

def poissonBlending(srcImgPath, dstImgPath, mixedGrad=True, linearCombination=False, product=0.5, weight=0.5):
    minColor = 0
    maxColor = 255
    poissonBlended = []
    naiveBlended = []
    srcRgbImg = imageToRgb(srcImgPath)
    dstRgbImg = imageToRgb(dstImgPath)
    for color in range(3):
        srcImg = np.asarray(srcRgbImg[color])
        dstImg = np.asarray(dstRgbImg[color])
        corner = topLeftCornerOfSrcOnDst(srcImg.shape, dstImg.shape)
        dstUnderSrc = cropDstUnderSrc(dstImg, corner, srcImg.shape)
        a, b, bShape = buildLinearSystem(srcImg, dstUnderSrc, mixedGrad, linearCombination, weight, product)
        x = solveLinearSystem(a, b, bShape, minColor, maxColor)
        constructMixedAndNaive(dstImg, srcImg, corner, x, bShape, poissonBlended, naiveBlended)
    mergeSaveShow(poissonBlended, 'poissonBlended.png', 'poissonBlended')
    mergeSaveShow(naiveBlended, 'naiveBlended.png', 'naiveBlended')

def main():
    poissonBlending('src_img/old_airplane3.jpg', 'dst_img/underwater5.jpg', True, False, 1, 0)

if __name__ == '__main__':
    main()
