# Haim Segal
import numpy as np
from scipy.sparse import csr_matrix
from PIL import Image
from pyamg.gallery import poisson
import pyamg
import matplotlib.pyplot as plt
from skimage.draw import polygon
np.seterr(divide='ignore', invalid='ignore')


def rgbToGrayMat(imagePath):
    grayImg = Image.open(imagePath).convert('L')
    return np.asarray(grayImg)

def polyMask(img, numOfPoints=100):
    plt.imshow(img, cmap='gray')
    plt.title('Select up to ' + str(numOfPoints) + ' points with the mouse', fontsize=16)
    pts = np.asarray(plt.ginput(numOfPoints, timeout=-1))
    row, col = polygon(tuple(pts[:, 1]), tuple(pts[:, 0]), img.shape)
    mask = np.zeros(img.shape)
    mask[row, col] = 1
    return mask, int(np.ceil(np.min(pts[:, 1]))), int(np.floor(np.max(pts[:, 1]))), int(np.ceil(np.min(pts[:, 0]))), int(np.floor(np.max(pts[:, 0])))

def cropIImgByLimits(src, minRow, maxRow, minCol, maxCol):
    r, g, b = src
    r = r[minRow: maxRow, minCol: maxCol]
    g = g[minRow: maxRow, minCol: maxCol]
    b = b[minRow: maxRow, minCol: maxCol]
    return r, g, b

def splitImageToRgb(imagePath):
    r, g, b = Image.Image.split(Image.open(imagePath))
    return np.asarray(r), np.asarray(g), np.asarray(b)

def topLeftCornerOfSrcOnDst(dst, srcShape, dstShape, horizontalBias=0, verticalBias=0):
    plt.imshow(dst, cmap='gray')
    plt.title('Select where you want to blend the source image un the destination image', fontsize=16)
    center = plt.ginput(1, timeout=-1)
    corner = [int(center[0][1]) - srcShape[0] // 2, int(center[0][0]) - srcShape[1] // 2]
    if corner[0] < 1:
        corner[0] = 1
    if corner[0] > dstShape[0] - srcShape[0] - 1:
        corner[0] = dstShape[0] - srcShape[0] - 1
    if corner[1] < 1:
        corner[1] = 1
    if corner[1] > dstShape[1] - srcShape[1] - 1:
        corner[1] = dstShape[1] - srcShape[1] - 1
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

def constructConstVector(mask, mixedGrad, linearCombination, dstUnderSrc, srcLaplacianed, srcShape):
    dstLaplacianed = laplacian(dstUnderSrc)
    if mixedGrad:
        b = np.reshape(linearCombination * mask * np.reshape(srcLaplacianed, srcShape) +
                       (1 - linearCombination) * mask * np.reshape(dstLaplacianed, srcShape) +
                       (mask - 1) * (- 1) * np.reshape(dstLaplacianed, srcShape), srcShape)
    else:
        b = np.reshape(mask * np.reshape(srcLaplacianed, srcShape) + (mask - 1) * (- 1) * np.reshape(dstLaplacianed, srcShape), srcShape)

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

def buildLinearSystem(mask, srcImg, dstUnderSrc, mixedGrad, linearCombination):
    srcLaplacianed = laplacian(srcImg)
    b = constructConstVector(mask, mixedGrad, linearCombination, dstUnderSrc, srcLaplacianed, srcImg.shape)
    a = constructCoefficientMat(b.shape)
    return a, b

def solveLinearSystem(a, b, bShape):
    multiLevel = pyamg.ruge_stuben_solver(csr_matrix(a))
    x = multiLevel.solve(b, tol=1e-10).reshape(bShape)
    x[x < 0] = 0
    x[x > 255] = 255
    return x

def blend(dst, patch, corner, patchShape, blended):
    mixed = dst.copy()
    mixed[corner[0]:corner[0] + patchShape[0], corner[1]:corner[1] + patchShape[1]] = patch
    blended.append(Image.fromarray(mixed))
    return blended

def poissonAndNaiveBlending(mask, corner, srcRgb, dstRgb, mixedGrad, linearCombination):
    poissonBlended = []
    naiveBlended = []
    for color in range(3):
        src = srcRgb[color]
        dst = dstRgb[color]
        dstUnderSrc = cropDstUnderSrc(dst, corner, src.shape)
        a, b = buildLinearSystem(mask, src, dstUnderSrc, mixedGrad, linearCombination)
        x = solveLinearSystem(a, b, b.shape)
        poissonBlended = blend(dst, x, (corner[0] + 1, corner[1] + 1), b.shape, poissonBlended)
        cropSrc = mask * src + (mask - 1) * (- 1) * dstUnderSrc
        naiveBlended = blend(dst, cropSrc, corner, src.shape, naiveBlended)
    return poissonBlended, naiveBlended

def mergeSaveShow(splittedImg, ImgName, ImgTitle):
    merged = Image.merge('RGB', tuple(splittedImg))
    merged.save(ImgName)
    merged.show(ImgTitle)

def poissonBlending(srcImgPath, dstImgPath, mixedGrad=True, linearCombination=0.8):
    srcGray = rgbToGrayMat(srcImgPath)
    mask, minRow, maxRow, minCol, maxCol = polyMask(srcGray)
    mask = mask[minRow: maxRow, minCol: maxCol]
    # plt.imshow(mask, cmap='gray')
    # plt.show()
    srcRgb = splitImageToRgb(srcImgPath)
    srcRgb = cropIImgByLimits(srcRgb, minRow, maxRow, minCol, maxCol)
    dstRgb = splitImageToRgb(dstImgPath)
    corner = topLeftCornerOfSrcOnDst(dstRgb[0], srcRgb[0].shape, dstRgb[0].shape)
    poissonBlended, naiveBlended = poissonAndNaiveBlending(mask, corner, srcRgb, dstRgb, mixedGrad, linearCombination)
    mergeSaveShow(poissonBlended, 'poissonBlended.png', 'Poisson Blended')
    mergeSaveShow(naiveBlended, 'naiveBlended.png', 'Naive Blended')

def main():
    poissonBlending('src_img/old_airplane3.jpg', 'dst_img/underwater5.jpg', True, 0.5)

if __name__ == '__main__':
    main()
