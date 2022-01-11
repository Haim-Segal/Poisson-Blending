# Haim Segal
import numpy as np
from scipy.sparse import csr_matrix
from PIL import Image
from pyamg.gallery import poisson
import pyamg
import time
import matplotlib.pyplot as plt
from matplotlib.path import Path
from skimage.draw import draw, polygon
np.seterr(divide='ignore', invalid='ignore')


def polyMask(img, numOfPoints=100):
    plt.imshow(img, cmap='gray')
    plt.title('Select up to ' + str(numOfPoints) + ' points with the mouse', fontsize=16)
    pts = np.asarray(plt.ginput(numOfPoints, timeout=-1))
    rr, cc = polygon(tuple(pts[:, 1]), tuple(pts[:, 0]), img.shape)
    mask = np.zeros(img.shape, dtype=np.double)
    mask[rr, cc] = 1
    return mask

def rgbToGrayMat(imagePath):
    grayImg = Image.open(imagePath).convert('L')
    return np.asarray(grayImg)


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

def poissonAndNaiveBlending(mask, corner, srcRgb, dstRgb, mixedGrad, linearCombination, weight, biggerConst):
    poissonBlended = []
    naiveBlended = []
    for color in range(3):
        src = srcRgb[color] * mask
        src = srcRgb[color]
        # plt.figure(1)
        # plt.imshow(src, cmap='gray')
        # plt.figure(2)
        # plt.imshow(src * mask)
        # plt.show()
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
    srcGray = rgbToGrayMat(srcImgPath)
    # plt.imshow(srcGray, cmap='gray')
    # plt.show()
    mask = polyMask(srcGray)
    plt.imshow(mask, cmap='gray')
    plt.show()
    srcRgb = splitImageToRgb(srcImgPath)
    dstRgb = splitImageToRgb(dstImgPath)
    corner = topLeftCornerOfSrcOnDst(srcRgb[0].shape, dstRgb[0].shape)
    poissonBlended, naiveBlended = poissonAndNaiveBlending(mask, corner, srcRgb, dstRgb, mixedGrad, linearCombination, weight, biggerConst)
    mergeSaveShow(poissonBlended, 'poissonBlended.png', 'Poisson Blended')
    mergeSaveShow(naiveBlended, 'naiveBlended.png', 'Naive Blended')

def tellme(s):
    plt.title(s, fontsize=16)
    # plt.draw()


def main():
    poissonBlending('src_img/old_airplane3.jpg', 'dst_img/underwater5.jpg', True, 0.8, 1, 0.8)

    # img = 0.5*np.ones((500, 500, 3), dtype=np.double)
    #
    # mask = polyMask(img)
    # plt.imshow(mask, vmin=0, vmax=1, cmap='gray')
    # plt.show()
















    #
    # poly = np.array((
    #     (300, 300),
    #     (480, 320),
    #     (380, 430),
    #     (220, 590),
    #     (300, 300),
    # ))
    # rr, cc = polygon(poly[:, 0], poly[:, 1], img.shape)
    # img[rr, cc, :] = 1
    # plt.imshow(img)
    # plt.show()


















    #
    # nx, ny = 100, 100
    # poly_verts = [(2, 2), (50, 10), (86, 92), (20, 23), (10, 10)]
    #
    # # Create vertex coordinates for each grid cell...
    # # (<0,0> is at the top left of the grid in this system)
    # x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    # x, y = x.flatten(), y.flatten()
    #
    # points = np.vstack((x, y)).T
    #
    # path = Path(poly_verts)
    # grid = path.contains_points(points)
    # grid = grid.reshape((ny, nx))
    # mask = np.asarray(grid)
    #
    # # print(grid)
    # # print('type(grid) =', type(grid))
    # # print('grid.shape =', grid.shape)
    # # plt.plot(grid)
    # # plt.imshow(mask, cmap='gray')
    # plt.figure()
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # pts = []
    # numOfPoints = 3
    # while len(pts) < numOfPoints:
    #     tellme('Select ' + str(numOfPoints) + ' corners with mouse')
    #     pts = np.asarray(plt.ginput(numOfPoints, timeout=-1))
    #     if len(pts) < numOfPoints:
    #         tellme('Too few points, starting over')
    #         time.sleep(1)  # Wait a second
    # ph = plt.fill(pts[:, 0], pts[:, 1], 'k', lw=2)
    # print('type(ph[0]) =', type(ph[0]))
    #
    # plt.show()
    #
    # row, col = draw.polygon((100, 200, 800), (100, 700, 400))














    # plt.figure()
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    #
    # tellme('You will define a triangle, click to begin')
    #
    # plt.waitforbuttonpress()
    #
    # while True:
    #     pts = []
    #     numOfPoints = 3
    #     while len(pts) < numOfPoints:
    #         tellme('Select ' + str(numOfPoints) + ' corners with mouse')
    #         pts = np.asarray(plt.ginput(numOfPoints, timeout=-1))
    #         if len(pts) < numOfPoints:
    #             tellme('Too few points, starting over')
    #             time.sleep(1)  # Wait a second
    #     ph = plt.fill(pts[:, 0], pts[:, 1], 'k', lw=2)
    #     tellme('Happy? Key click for yes, mouse click for no')
    #
    #     if plt.waitforbuttonpress():
    #         break
    #
    #     # Get rid of fill
    #     for p in ph:
    #         p.remove()


if __name__ == '__main__':
    main()
