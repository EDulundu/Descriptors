import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

# fractal dimension calculation using box-counting method
class FractalDimension:

    @staticmethod
    def compute_descriptor(threshed_image, black=True, debug=False):

        imgy, imgx = threshed_image.shape

        min_axis = min(imgx, imgy)

        if black:
            theColor = 0
        else:
            theColor = 255

        gx = []
        gy = []

        for ib in (8, 4, 2):

            box_size = ib
            box_x = int(imgx // box_size)
            box_y = int(imgy // box_size)

            boxCount = 0
            for y in range(box_y):
                for x in range(box_x):

                    # if there are any pixels in the box then increase box count
                    foundPixel = False
                    for ky in range(box_size):
                        for kx in range(box_size):

                            if thresh[box_size * y + ky, box_size * x + kx] == theColor:
                                foundPixel = True
                                boxCount += 1
                                break

                        if foundPixel:
                            break

            r = 1.0 / (min_axis / box_size)
            gx.append(1.0 / r)
            gy.append(boxCount)

        if debug:
            plt.plot(np.log(gx), np.log(gy))
            plt.show()

        return np.polyfit(np.log(gx), np.log(gy), 1)[0]

    @staticmethod
    def compute_descriptor_2(threshed_image, threshold=0.9):

        # Only for 2d image
        assert (len(threshed_image.shape) == 2)

        # From https://github.com/rougier/numpy-100 (#87)
        def boxcount(threshed_image, k):
            S = np.add.reduceat(
                np.add.reduceat(threshed_image, np.arange(0, threshed_image.shape[0], k), axis=0),
                np.arange(0, threshed_image.shape[1], k), axis=1)

            # We count non-empty (0) and non-full boxes (k*k)
            return len(np.where((S > 0) & (S < k * k))[0])

        # Transform Z into a binary array
        threshed_image = (threshed_image < threshold)

        # Minimal dimension of image
        p = min(threshed_image.shape)

        # Greatest power of 2 less than or equal to p
        n = 2 ** np.floor(np.log(p) / np.log(2))

        # Extract the exponent
        n = int(np.log(n) / np.log(2))

        # Build successive box sizes (from 2**n down to 2**1)
        sizes = 2 ** np.arange(n, 1, -1)

        # Actual box counting with decreasing size
        counts = []
        for size in sizes:
            counts.append(boxcount(threshed_image, size))

        # Fit the successive log(sizes) with log (counts)
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)

        return -coeffs[0]

if __name__ == '__main__':

    # I = rgb2gray(scipy.misc.imread("triangle.jpg") / 256)
    # print("Minkowski–Bouligand dimension (computed): ", fractal_dimension(I))
    # print("Haussdorf dimension (theoretical):        ", (np.log(3) / np.log(2)))

    read = "data/SCUT-FBP-"
    result = "/home/emre/Desktop/Feature Vectors/FRACTAL/SCUT-FBP-"

    for i in range(1, 501):

        image = cv2.imread(read + str(i) + ".jpg")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_OTSU)

        feature = FractalDimension.compute_descriptor(thresh, black=True, debug=False)

        fo = open(result + str(i) + ".txt", "w+")
        fo.write(str(feature) + "\n")
        fo.close()

        print(i, "islendi")