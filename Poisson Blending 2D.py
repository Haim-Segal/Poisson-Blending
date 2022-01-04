# Haim Segal
import numpy as np
from scipy.sparse import csr_matrix
from PIL import Image
from pyamg.gallery import poisson
import pyamg


def splitImageToRgb(imagePath):
    r, g, b = Image.Image.split(Image.open(imagePath))
    return np.asarray(r), np.asarray(g), np.asarray(b)

def topLeftCornerOfSrcOnDst(srcShape, dstShape, horizontalBias=-200, verticalBias=-350):
    center = (dstShape[0] // 2 + horizontalBias, dstShape[1] // 2 + verticalBias)
    corner = (center[0] - srcShape[0] // 2, center[1] - srcShape[1] // 2)
    return corner

def cropDstUnderSrc(dstImg, corner, srcShape):
    dstUnderSrc = dstImg[
                   corner[0]:corner[0] + srcShape[0],
                   corner[1]:corner[1] + srcShape[1]]
    return dstUnderSrc

def laplacian(array):
    return (poisson(array.shape, format='csr') * csr_matrix(array.flatten()).transpose()).toarray()

def construct_const_vector(mixedGrad, linearCombination, dstUnderSrc, srcLaplacianed, weight, srcShape, product):
    if mixedGrad:
        dstLaplacianed = laplacian(dstUnderSrc)
        if linearCombination:
            b = np.reshape(
                weight * srcLaplacianed + (1 - weight) * dstLaplacianed, srcShape)
        else:
            biggerLaplacian = abs(srcLaplacianed) >= product * abs(dstLaplacianed)
            b = np.reshape(biggerLaplacian * srcLaplacianed +
                           (1 - weight) * ~biggerLaplacian * dstLaplacianed +
                           weight * ~biggerLaplacian * srcLaplacianed, srcShape)
    else:
        b = np.reshape(srcLaplacianed, srcShape)
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

def solveLinearSystem(a, b, bShape):
    multi_level = pyamg.ruge_stuben_solver(csr_matrix(a))
    x = multi_level.solve(b, tol=1e-10).reshape(bShape)
    x[x < 0] = 0
    x[x > 255] = 255
    return x

def blend(dst, patch, corner, patchShape, blended):
    mixed = dst.copy()
    mixed[corner[0]:corner[0] + patchShape[0], corner[1]:corner[1] + patchShape[1]] = patch
    blended.append(Image.fromarray(mixed))
    return blended

def possionAndNaiveBlending(corner, srcRgb, dstRgb, mixedGrad, linearCombination, weight, product):
    poissonBlended = []
    naiveBlended = []
    for color in range(3):
        src = srcRgb[color]
        dst = dstRgb[color]
        dstUnderSrc = cropDstUnderSrc(dst, corner, src.shape)
        a, b, bShape = buildLinearSystem(src, dstUnderSrc, mixedGrad, linearCombination, weight, product)
        x = solveLinearSystem(a, b, bShape)
        poissonBlended = blend(dst, x, (corner[0] + 1, corner[1] + 1), bShape, poissonBlended)
        naiveBlended = blend(dst, src, corner, src.shape, naiveBlended)
    return poissonBlended, naiveBlended

def mergeSaveShow(splittedImg, ImgName, ImgTitle):
    merged = Image.merge('RGB', tuple(splittedImg))
    merged.save(ImgName)
    merged.show(ImgTitle)

def poissonBlending(srcImgPath, dstImgPath, mixedGrad=True, linearCombination=False, product=0.5, weight=0.5):
    srcRgb = splitImageToRgb(srcImgPath)
    dstRgb = splitImageToRgb(dstImgPath)
    corner = topLeftCornerOfSrcOnDst(srcRgb[0].shape, dstRgb[0].shape)
    poissonBlended, naiveBlended = possionAndNaiveBlending(corner, srcRgb, dstRgb, mixedGrad, linearCombination, weight, product)
    mergeSaveShow(poissonBlended, 'poissonBlended.png', 'Poisson Blended')
    mergeSaveShow(naiveBlended, 'naiveBlended.png', 'Naive Blended')

def main():
    poissonBlending('src_img/old_airplane3.jpg', 'dst_img/underwater5.jpg')

if __name__ == '__main__':
    main()
