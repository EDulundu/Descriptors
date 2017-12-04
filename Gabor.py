import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io, scipy.misc, scipy.signal

class Gabor:

    # opencv implementation
    # kernel size elle girilir.
    @staticmethod
    def opencv_gabor_kernel(ksize, sigma, theta, lambd, gamma, psi, ktype):

        sigma_x = sigma
        sigma_y = sigma / gamma

        cos = np.cos(theta)
        sin = np.sin(theta)

        xmax = ksize[0] // 2
        ymax = ksize[1] // 2

        xmin = -xmax
        ymin = -ymax

        real = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=ktype)

        scale = 1
        ex = -0.5 / (sigma_x * sigma_x)
        ey = -0.5 / (sigma_y * sigma_y)
        cscale = np.pi * 2 / lambd

        for y in range(ymin, ymax+1):
            for x in range(xmin, xmax+1):

                xr = x * cos + y * sin
                yr = -x * sin + y * cos

                v = scale * np.exp(ex * xr * xr + ey * yr * yr) * np.cos(cscale * xr + psi)
                real[ymax - y, xmax - x] = float(v)

        return real

    # wikipedia implementation
    # kernel size otomatik parametreler tarafından oluşur.
    @staticmethod
    def wiki_gabor_kernel(sigma, theta, Lambda, psi, gamma):

        sigma_x = sigma
        sigma_y = float(sigma) / gamma

        # Bounding box
        nstds = 3  # Number of standard deviation sigma
        xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
        xmax = np.ceil(max(1, xmax))
        ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
        ymax = np.ceil(max(1, ymax))
        xmin = -xmax
        ymin = -ymax
        (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

        # Rotation
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)

        real = np.exp(-.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)

        return real

    # benim gaborum. temel parametreler theta, frekans ve sigma
    #
    # theta -> tespit edilecek ayrıtların yönü. filtrenin yönünü belirler.
    # frekans -> filtredeki ayrıtların yada gabor dalgacıkların frekansını belirler. dalgacıkların sıklığıda diyebiliriz.
    # sigma -> gaus fonksiyonundaki x ve y yönündeki standart sapmadır. sigma ayrıca dalgacıkların çapını belirler. scale olarakda gecmektedir.
    # ksize -> kernel matrixin boyutu.
    # gamma -> gabor dalgacıkların elipsliğini ayarlar. (0.2 - 1) genellikle aralıgındadır.
    # bandwidth -> genelde 1 dir. (0.4 - 2.5) aralığındadır.
    # psi -> 0 dır.
    @staticmethod
    def my_gabor_kernel(theta=0, frequency=1, sigma_x=None, sigma_y=None, ksize=None, bandwidth=1.0, gamma=1.0, psi=0):

        if sigma_x is None:
            sigma_x = Gabor.sigma_prefactor(bandwidth) / frequency
        if sigma_y is None:
            sigma_y = Gabor.sigma_prefactor(bandwidth) / frequency

        nstds = 3
        cos = np.cos(theta)
        sin = np.sin(theta)

        if ksize is None:
            xmax = np.ceil(max(abs(nstds * sigma_x * cos), abs(nstds * sigma_y * sin)))
            ymax = np.ceil(max(abs(nstds * sigma_x * sin), abs(nstds * sigma_y * cos)))
        else:
            ymax, xmax = ksize
            if xmax > 0:
                xmax = (xmax - 1) // 2
            if ymax > 0:
                ymax = (ymax - 1) // 2

        xmin = -xmax
        ymin = -ymax

        (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

        x_theta = x * cos + y * sin
        y_theta = -x * sin + y * cos

        return np.exp(-.5 * (x_theta**2 / sigma_x**2 + gamma * gamma * y_theta**2 / sigma_y**2)) * \
                        (np.cos(2 * np.pi * frequency * x_theta + psi))

    @staticmethod
    def sigma_prefactor(bandwidth):

        return 1.0 / np.pi * np.sqrt(np.log(2.0) / 2.0) * (2.0**bandwidth + 1.0) / (2.0**bandwidth - 1.0)

    @staticmethod
    def normalize(kernel):

        ppos = np.where(kernel > 0)
        pos = kernel[ppos].sum()

        pneg = np.where(kernel < 0)
        neg = kernel[pneg].sum()

        meansum = (pos - neg) / 2
        if meansum > 0:
            pos = pos / meansum
            neg = -neg / meansum

        kernel[pneg] = pos * kernel[pneg]
        kernel[ppos] = neg * kernel[ppos]

        return kernel

def convolve(image, kernel):

    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    pad = (kW - 1) // 2
    image = np.lib.pad(image, pad, "symmetric")
    output = np.zeros((iH, iW), dtype="float32")

    for y in range(pad, iH + pad):
        for x in range(pad, iW + pad):

            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            k = (roi * kernel).sum()
            output[y - pad, x - pad] = k

    return output

def create_filter_bank(orientations, scales):

    kernels = []
    for theta in orientations:
        for sigma in scales:
            kernel = Gabor.my_gabor_kernel(theta=theta, frequency=0.1, sigma_x=sigma, sigma_y=sigma, ksize=(9, 9))
            normalized_kernel = Gabor.normalize(kernel)
            kernels.append(normalized_kernel)

    return kernels

def extract_feature_all_images(kernels):

    index = 1
    last_index = 501

    read = "data/SCUT-FBP-"
    result = "/home/emre/Desktop/Feature Vectors/GABOR/SCUT-FBP-"

    for i in range(index, last_index):
        path = read + str(i) + ".jpg"
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        featureVector = []
        for kernel in kernels:
            filtered = scipy.signal.convolve2d(gray, kernel, mode='same', boundary='symm')
            featureVector.append(np.mean(filtered))
            featureVector.append(np.std(filtered))

        fo = open(result + str(i) + ".txt", "w+")
        for j in range(len(featureVector)):
            fo.write(str(featureVector[j]) + "\n")
        fo.close()

        print(path, "islendi")

if __name__ == '__main__':

    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    scales = [2.0, 2.5, 3.0]
    filterBank = create_filter_bank(orientations, scales)

    """
    row = 4
    col = 3

    fig, axes = plt.subplots(nrows=row, ncols=col, figsize=(15, 10))
    fig.suptitle("Gabor filters")

    plt.gray()

    index = 0
    for ax_row in range(row):
        for ax_col in range(col):
            axes[ax_row, ax_col].imshow(filterBank[index])
            index += 1

    plt.show()
    """

    ################################################################################################

    start = time.time()

    extract_feature_all_images(filterBank)

    print("Execution time: ", time.time() - start)