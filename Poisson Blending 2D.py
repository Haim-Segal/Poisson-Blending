# Haim Segal
import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix
import pyamg
from pyamg.gallery import poisson
import matplotlib.pyplot as plt
from skimage.draw import polygon

np.seterr(divide='ignore', invalid='ignore')

def getImagePathFromUser(srcOrDst):
    print('Open ' + str(srcOrDst) + ' image')
    tk.Tk().withdraw()
    return filedialog.askopenfilename()

def rgbToGrayMat(imagePath):
    grayImg = Image.open(imagePath).convert('L')
    return np.asarray(grayImg)

def polyMask(imgPath, numOfPoints=100):
    img = rgbToGrayMat(imgPath)
    plt.imshow(img, cmap='gray')
    plt.title('Create polygon capturing the area you want to blend')
    pts = np.asarray(plt.ginput(numOfPoints, timeout=-1))
    print('pts = ', pts)
    if len(pts) < 3:
        mask = np.ones(img.shape)
        minRow, minCol = (0, 0)
        maxRow, maxCol = img.shape
    else:
        insidePolyRow, insidePolyCol = polygon(tuple(pts[:, 1]), tuple(pts[:, 0]), img.shape)
        mask = np.zeros(img.shape)
        mask[insidePolyRow, insidePolyCol] = 1
        minRow, minCol = np.floor(np.min(pts, axis=0)).astype(int)
        maxRow, maxCol = np.ceil(np.max(pts, axis=0)).astype(int)
        # mask = mask[minRow: maxRow, minCol: maxCol]
    return mask, minRow, maxRow, minCol, maxCol

def splitImageToRgb(imagePath):
    r, g, b = Image.Image.split(Image.open(imagePath))
    return np.asarray(r), np.asarray(g), np.asarray(b)

def cropImgByLimits(src, minRow, maxRow, minCol, maxCol):
    r, g, b = src
    r = r[minRow: maxRow, minCol: maxCol]
    g = g[minRow: maxRow, minCol: maxCol]
    b = b[minRow: maxRow, minCol: maxCol]
    return r, g, b

def topLeftCornerOfSrcOnDst(dstImgPath, srcShape):
    grayDst = rgbToGrayMat(dstImgPath)
    plt.imshow(grayDst, cmap='gray')
    plt.title('where you want to blend it..?')
    center = plt.ginput(1, timeout=-1)
    if len(center) < 1:
        center = [[grayDst.shape[1] // 2, grayDst.shape[0] // 2]]
    corner = [int(center[0][1]) - srcShape[0] // 2, int(center[0][0]) - srcShape[1] // 2]
    if corner[0] < 1:
        corner[0] = 1
    if corner[0] > grayDst.shape[0] - srcShape[0] - 1:
        corner[0] = grayDst.shape[0] - srcShape[0] - 1
    if corner[1] < 1:
        corner[1] = 1
    if corner[1] > grayDst.shape[1] - srcShape[1] - 1:
        corner[1] = grayDst.shape[1] - srcShape[1] - 1
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

def constructConstVector(mask, mixedGrad, dstUnderSrc, srcLaplacianed, srcShape):
    dstLaplacianed = laplacian(dstUnderSrc)
    b = np.reshape((1 - mixedGrad) * mask * np.reshape(srcLaplacianed, srcShape) +
                   mixedGrad * mask * np.reshape(dstLaplacianed, srcShape) +
                   (mask - 1) * (- 1) * np.reshape(dstLaplacianed, srcShape), srcShape)
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

def mergeSaveShow(splittedImg, ImgTitle):
    merged = Image.merge('RGB', tuple(splittedImg))
    merged.save(ImgTitle + '.png')
    merged.show(ImgTitle)

def main():
    srcImgPath = getImagePathFromUser('source')
    mask, *maskLimits = polyMask(srcImgPath)
    print('*maskLimits = ', *maskLimits)
    srcRgb = splitImageToRgb(srcImgPath)
    # srcRgbCropped = cropImgByLimits(srcRgb, *maskLimits)
    srcRgbCropped = srcRgb
    dstImgPath = getImagePathFromUser('destination')
    dstRgb = splitImageToRgb(dstImgPath)
    corner = topLeftCornerOfSrcOnDst(dstImgPath, srcRgbCropped[0].shape)
    poissonBlended, naiveBlended = poissonAndNaiveBlending(mask, corner, srcRgbCropped, dstRgb, 0.3)
    mergeSaveShow(poissonBlended, 'Poisson Blended')
    mergeSaveShow(naiveBlended, 'Naive Blended')

if __name__ == '__main__':
    main()
