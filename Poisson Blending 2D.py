import tkinter
from PIL import Image
import numpy as np
from scipy.sparse import csr_matrix
from pyamg import ruge_stuben_solver
from pyamg.gallery import poisson
import matplotlib.pyplot as plt
from skimage.draw import polygon


def getImagePathFromUser(srcOrDst):
    tkinter.Tk().withdraw()
    return tkinter.filedialog.askopenfilename(title='Open ' + str(srcOrDst) + ' image')


def rgbToGrayMat(imgPth):
    grayImg = Image.open(imgPth).convert('L')
    return np.asarray(grayImg)


def polyMask(imgPth, numOfPts=100):
    img = rgbToGrayMat(imgPth)
    plt.figure('source image')
    plt.title('Inscribe the region you would like to blend inside a polygon')
    plt.imshow(img, cmap='gray')
    pts = np.asarray(plt.ginput(numOfPts, timeout=-1))
    plt.close('source image')
    if len(pts) < 3:
        mask = np.ones(img.shape)
        minRow, minCol = (0, 0)
        maxRow, maxCol = img.shape
    else:
        pts = np.fliplr(pts)
        inPolyRow, inPolyCol = polygon(tuple(pts[:, 0]), tuple(pts[:, 1]), img.shape)
        mask = np.zeros(img.shape)
        mask[inPolyRow, inPolyCol] = 1
        minRow, minCol = np.max(np.vstack([np.floor(np.min(pts, axis=0)).astype(int).reshape((1, 2)), [0, 0]]), axis=0)
        maxRow, maxCol = np.min(np.vstack([np.ceil(np.max(pts, axis=0)).astype(int).reshape((1, 2)), mask.shape]),
                                axis=0)
        mask = mask[minRow: maxRow, minCol: maxCol]
    return mask, minRow, maxRow, minCol, maxCol


def splitImageToRgb(imgPth):
    r, g, b = Image.Image.split(Image.open(imgPth))
    return np.asarray(r), np.asarray(g), np.asarray(b)


def cropImgByLimits(src, minRow, maxRow, minCol, maxCol):
    r, g, b = src
    r = r[minRow: maxRow, minCol: maxCol]
    g = g[minRow: maxRow, minCol: maxCol]
    b = b[minRow: maxRow, minCol: maxCol]
    return r, g, b


def topLeftCornerOfSrcOnDst(dstImgPth, srcShp):
    grayDst = rgbToGrayMat(dstImgPth)
    plt.figure('destination image')
    plt.title('Where would you like to blend it..?')
    plt.imshow(grayDst, cmap='gray')
    center = np.asarray(plt.ginput(2, -1, True)).astype(int)
    plt.close('destination image')
    if len(center) < 1:
        center = np.asarray([[grayDst.shape[1] // 2, grayDst.shape[0] // 2]]).astype(int)
    elif len(center) > 1:
        center = np.asarray([center[0]])
    corner = [center[0][1] - srcShp[0] // 2, center[0][0] - srcShp[1] // 2]
    if corner[0] < 1:
        corner[0] = 1
    if corner[0] > grayDst.shape[0] - srcShp[0] - 1:
        corner[0] = grayDst.shape[0] - srcShp[0] - 1
    if corner[1] < 1:
        corner[1] = 1
    if corner[1] > grayDst.shape[1] - srcShp[1] - 1:
        corner[1] = grayDst.shape[1] - srcShp[1] - 1
    return corner


def cropDstUnderSrc(dstImg, corner, srcShp):
    dstUnderSrc = dstImg[
                  corner[0]:corner[0] + srcShp[0],
                  corner[1]:corner[1] + srcShp[1]]
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


def constructConstVector(mask, mixedGrad, dstUnderSrc, srcLaplacianed, srcShp):
    dstLaplacianed = laplacian(dstUnderSrc)
    b = np.reshape((1 - mixedGrad) * mask * np.reshape(srcLaplacianed, srcShp) +
                   mixedGrad * mask * np.reshape(dstLaplacianed, srcShp) +
                   (mask - 1) * (- 1) * np.reshape(dstLaplacianed, srcShp), srcShp)
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


def buildLinearSystem(mask, srcImg, dstUnderSrc, mixedGrad):
    srcLaplacianed = laplacian(srcImg)
    b = constructConstVector(mask, mixedGrad, dstUnderSrc, srcLaplacianed, srcImg.shape)
    a = constructCoefficientMat(b.shape)
    return a, b


def solveLinearSystem(a, b, bShape):
    multiLevel = ruge_stuben_solver(csr_matrix(a))
    x = np.reshape((multiLevel.solve(b.flatten(), tol=1e-10)), bShape)
    x[x < 0] = 0
    x[x > 255] = 255
    return x


def blend(dst, patch, corner, patchShape, blended):
    mixed = dst.copy()
    mixed[corner[0]:corner[0] + patchShape[0], corner[1]:corner[1] + patchShape[1]] = patch
    blended.append(Image.fromarray(mixed))
    return blended


def poissonAndNaiveBlending(mask, corner, srcRgb, dstRgb, mixedGrad):
    poissonBlended = []
    naiveBlended = []
    for color in range(3):
        src = srcRgb[color]
        dst = dstRgb[color]
        dstUnderSrc = cropDstUnderSrc(dst, corner, src.shape)
        a, b = buildLinearSystem(mask, src, dstUnderSrc, mixedGrad)
        x = solveLinearSystem(a, b, b.shape)
        poissonBlended = blend(dst, x, (corner[0] + 1, corner[1] + 1), b.shape, poissonBlended)
        cropSrc = mask * src + (mask - 1) * (- 1) * dstUnderSrc
        naiveBlended = blend(dst, cropSrc, corner, src.shape, naiveBlended)
    return poissonBlended, naiveBlended


def mergeSaveShow(splittedImg, ImgTtl):
    merged = Image.merge('RGB', tuple(splittedImg))
    merged.save(ImgTtl + '.png')
    merged.show(ImgTtl)


def main():
    srcImgPth = getImagePathFromUser('source')
    mask, *maskLimits = polyMask(srcImgPth)
    srcRgb = splitImageToRgb(srcImgPth)
    srcRgbCropped = cropImgByLimits(srcRgb, *maskLimits)
    dstImgPth = getImagePathFromUser('destination')
    dstRgb = splitImageToRgb(dstImgPth)
    corner = topLeftCornerOfSrcOnDst(dstImgPth, srcRgbCropped[0].shape)
    poissonBlended, naiveBlended = poissonAndNaiveBlending(mask, corner, srcRgbCropped, dstRgb, 0.3)
    mergeSaveShow(naiveBlended, 'Naive Blended')
    mergeSaveShow(poissonBlended, 'Poisson Blended')


if __name__ == '__main__':
    main()