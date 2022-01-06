# Haim Segal
import numpy as np
from scipy.sparse import csr_matrix
from PIL import Image
from pyamg.gallery import poisson
import pyamg
np.seterr(divide='ignore', invalid='ignore')

def splitImageToRgb(imagePath):
    r, g, b = Image.Image.split(Image.open(imagePath))
    return np.asarray(r), np.asarray(g), np.asarray(b)

def topLeftCornerOfSrcOnDst(srcShape, dstShape, horizontalBias=+50, verticalBias=+350):
    center = (dstShape[0] // 2 + horizontalBias, dstShape[1] // 2 + verticalBias)
    corner = (center[0] - srcShape[0] // 2, center[1] - srcShape[1] // 2)
    return corner

def cropDstUnderSrc(dstImg, corner, srcShape):
    dstUnderSrc = dstImg[
                  corner[0]:corner[0] + srcShape[0],
                  corner[1]:corner[1] + srcShape[1]]
    return dstUnderSrc

def laplacian(array):
    return poisson(array.shape, format='csr') * csr_matrix(array.flatten()).transpose().toarray()

def setBoundaryCondition(b, dstUnderSrc):
    b[1, :] = dstUnderSrc[1, :]
    b[-2, :] = dstUnderSrc[-2, :]
    b[:, 1] = dstUnderSrc[:, 1]
    b[:, -2] = dstUnderSrc[:, -2]
    b = b[1:-1, 1: -1]
    return b

def constructConstVector(mixedGrad, linearCombination, dstUnderSrc, srcLaplacianed, weight, srcShape, biggerConst):
    if mixedGrad:
        dstLaplacianed = laplacian(dstUnderSrc)
        if linearCombination:
            b = np.reshape(
                linearCombination * srcLaplacianed + (1 - linearCombination) * dstLaplacianed, srcShape)
        else:
            biggerLaplacian = abs(srcLaplacianed) >= biggerConst * abs(dstLaplacianed)
            b = np.reshape(biggerLaplacian * srcLaplacianed +
                           (1 - weight) * ~biggerLaplacian * dstLaplacianed +
                           weight * ~biggerLaplacian * srcLaplacianed, srcShape)
    else:
        b = np.reshape(srcLaplacianed, srcShape)
    return setBoundaryCondition(b, dstUnderSrc)

def fixCoeffUnderBoundaryCondition(coeff, shape):
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

def constructCoefficientMat(shape):
    a = poisson(shape, format='lil')
    a = fixCoeffUnderBoundaryCondition(a, shape)
    return a

def buildLinearSystem(srcImg, dstUnderSrc, mixedGrad, linearCombination, weight, biggerConst):
    srcLaplacianed = laplacian(srcImg)
    b = constructConstVector(mixedGrad, linearCombination, dstUnderSrc, srcLaplacianed, weight, srcImg.shape, biggerConst)
    a = constructCoefficientMat(b.shape)
    return a, b

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

def poissonAndNaiveBlending(corner, srcRgb, dstRgb, mixedGrad, linearCombination, weight, biggerConst):
    poissonBlended = []
    naiveBlended = []
    for color in range(3):
        src = srcRgb[color]
        dst = dstRgb[color]
        dstUnderSrc = cropDstUnderSrc(dst, corner, src.shape)
        a, b = buildLinearSystem(src, dstUnderSrc, mixedGrad, linearCombination, weight, biggerConst)
        x = solveLinearSystem(a, b, b.shape)
        poissonBlended = blend(dst, x, (corner[0] + 1, corner[1] + 1), b.shape, poissonBlended)
        naiveBlended = blend(dst, src, corner, src.shape, naiveBlended)
    return poissonBlended, naiveBlended

def mergeSaveShow(splittedImg, ImgName, ImgTitle):
    merged = Image.merge('RGB', tuple(splittedImg))
    merged.save(ImgName)
    merged.show(ImgTitle)

def poissonBlending(srcImgPath, dstImgPath, mixedGrad=True, linearCombination=0.5, biggerConst=0.5, weight=0.9):
    srcRgb = splitImageToRgb(srcImgPath)
    dstRgb = splitImageToRgb(dstImgPath)
    corner = topLeftCornerOfSrcOnDst(srcRgb[0].shape, dstRgb[0].shape)
    poissonBlended, naiveBlended = poissonAndNaiveBlending(corner, srcRgb, dstRgb, mixedGrad, linearCombination, weight, biggerConst)
    mergeSaveShow(poissonBlended, 'poissonBlended.png', 'Poisson Blended')
    mergeSaveShow(naiveBlended, 'naiveBlended.png', 'Naive Blended')

def main():
    poissonBlending('src_img/old_airplane3.jpg', 'dst_img/underwater5.jpg', True, 0.8, 1, 0.8)
    # a = np.array(np.ones((3, 3)))
    # b = np.array(2 * np.ones((3, 3)))
    # c = abs(a) < abs(b)
    # d = a / b
    # e = (a * abs(a) + b * abs(b)) / (abs(a) + abs(b))

    # print('a = ', a)
    # print('b = ', b)
    # print('c = ', c)
    # print('d = ', d)
    # print('e = ', e)

if __name__ == '__main__':
    main()
