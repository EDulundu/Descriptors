import cv2
import time
import numpy as np

class Hog:

    """
        0. (optional) global image normalisation
        1. convert the image to grey scale
        2. compute the gradient image in x and y
        3. compute magnitude and orientation
        4. compute gradient histograms with bi-linear interpolation.
        5. normalize the histograms
        6. convert to 1-D descriptor vector

        Dalal and Trigs
    """

    def __init__(self, block_size=(2, 2), cell_size=(8, 8), vector_size=3):

        """
        Histogram of oriented gradient constructor.
        :param block_size: bloğun içindeki hücre sayısı.
        :param cell_size:  hücre içerisindeki piksel sayısı
        :param vector_size: özellik vectörünün uzunluğunu belirler.
        """

        self.block_size = block_size
        self.cell_size = cell_size
        self.vector_size = vector_size
        self.nbins = 9

    def gradient(self, image):

        """
        verilen gri bir resimin gradyanını alır
        :param image: gri resim
        :return: gx, gy, x ve y yönündeki gradyanlar.
        """
        # sobel operatörü ile x ve yönünde sırayla türev alınır.
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)

        return gx, gy

    def magnitude_orientation(self, gx, gy):

        """
        gradyanı alınan resmin magnitude'nu hesaplar.
        0-180 arası her bir pikselin açısını hesaplar.
        :param gx: x yönündeki gradyan
        :param gy: y yönündeki gradyan
        :return: magnitude ve açı 2D dizisi.
        """

        # (x^2  + y^2)^0.5 magnitude alınır. açı hesaplanır. mod 180 ile açı 0-180 aralığına çekilir.
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        orientation = (np.arctan2(gy, gx) * 180.0 / np.pi) % 180

        return magnitude, orientation

    def build_histogram(self, magnitude, orientation):

        size_y, size_x = magnitude.shape
        cell_size_y, cell_size_x = self.cell_size
        bin_step = 180 // self.nbins

        # checking that the cell size are even
        if cell_size_x % 2 != 0:
            cell_size_x += 1
            print("WARNING: the cell_size must be even, incrementing cell_size_x of 1")

        if cell_size_y % 2 != 0:
            cell_size_y += 1
            print("WARNING: the cell_size must be even, incrementing cell_size_y of 1")

        # Consider only the right part of the image
        # (if the rest doesn't fill a whole cell, just drop it)
        size_x -= size_x % cell_size_x
        size_y -= size_y % cell_size_y

        # x ve y eksenindeki toplam cell sayıları hesaplanır.
        n_cells_x = size_x // cell_size_x
        n_cells_y = size_y // cell_size_y

        # sınır dışı kalan veriler göz ardı edilir.
        magnitude = magnitude[:size_y, :size_x]
        orientation = orientation[:size_y, :size_x]
        gradient_hist = np.zeros((n_cells_y, n_cells_x, self.nbins), dtype=float)

        # The angel that was 180 degree is made 0 degree.
        orientation[np.where(orientation // bin_step >= 9)] = 0

        for y in range(n_cells_y):

            start_y = y * cell_size_y
            end_y = (y + 1) * cell_size_y

            for x in range(n_cells_x):

                start_x = x * cell_size_x
                end_x = (x + 1) * cell_size_x

                # get the angel and mag of the cell
                cell_angle = abs(orientation[start_y:end_y, start_x:end_x])
                cell_mag = magnitude[start_y:end_y, start_x:end_x]
                bin_number = abs(cell_angle // bin_step)

                # binning process bi-linearly interpolation
                ratio_lower = (((bin_number + 1) * bin_step) - cell_angle) / 20
                ratio_upper = (cell_angle - (bin_number * bin_step)) / 20

                result_lower = cell_mag * ratio_lower  # bin <- gidecek mag sonucu
                result_upper = cell_mag * ratio_upper  # bin+1 <- gidecek mag sonucu

                # histogram addition
                for j in range(cell_size_y):
                    for i in range(cell_size_x):
                        number = int(bin_number[j, i])
                        gradient_hist[y, x, number % self.nbins] += result_lower[j, i]
                        gradient_hist[y, x, (number + 1) % self.nbins] += result_upper[j, i]

        by, bx = self.block_size

        # x ve y eksenindeki block size hesaplanır.
        n_blocksx = (n_cells_x - bx) + 1
        n_blocksy = (n_cells_y - by) + 1

        # l2 hys normalizasyon yapılır.
        normalized_histogram = self.normalizeBlock(gradient_hist, n_blocksy, n_blocksx, by, bx)

        box_histogram = self.combine_histogram(normalized_histogram, n_cells_y, n_cells_x, by, bx)

        return box_histogram

    def normalizeBlock(self, gradient_histogram, n_blocksy, n_blocksx, by, bx):

        normalized_blocks = np.zeros((n_blocksy, n_blocksx, by, bx, 9), dtype=float)

        # normalizasyon yaparken karesinin toplamımı yoksa toplamlarının karesimi tam anlamadım orayı.
        for x in range(n_blocksx):
            for y in range(n_blocksy):
                block = gradient_histogram[y:y + by, x:x + bx, :]
                out = np.clip(block / np.sqrt((block.ravel() ** 2).sum() + 1e-5), 0, 0.2)
                out /= np.sqrt((out.ravel() ** 2).sum() + 1e-5)
                normalized_blocks[y, x, :] = out

        return normalized_blocks

    def combine_histogram(self, normalized_block, n_cells_y, n_cells_x, by, bx):

        histogram = np.zeros((self.vector_size, self.vector_size, 9), dtype=float)

        step_y = n_cells_y // (self.vector_size + 1)  # 20
        step_x = n_cells_x // (self.vector_size + 1)  # 15

        for j in range(self.vector_size):

            start_y = j * step_y
            end_y = start_y + 2 * step_y
            for i in range(self.vector_size):

                start_x = i * step_x
                end_x = start_x + 2 * step_x

                sum = np.zeros((9), dtype=float)
                for ky in range(start_y, end_y - 1):
                    for kx in range(start_x, end_x - 1):
                        for ly in range(by):
                            for lx in range(bx):
                                sum += normalized_block[ky, kx, ly, lx, :]

                histogram[j, i] = sum

        return histogram

    def compute(self, image):

        """
        Parametreleri verilen betimleyicinin hesaplandığı genel fonksiyon diğer
        tüm fonksiyonlar bunun altında çağırılır.
        :param image: gri resim.
        :return: normalize edilmiş tek boyutlu float vektör.
        """
        # x ve y yönünde gradient alır.
        gx, gy = self.gradient(image)

        # magnitude ve orientation hesaplanır. açı 0-180 arasıdır.
        magnitude, orientation = self.magnitude_orientation(gx, gy)

        # gradient histogramını hesaplar ve normalize edip return eder.
        return self.build_histogram(magnitude, orientation).ravel()

def extract_feature_all_images():

    index = 1
    last_index = 501
    HOGDescriptor = Hog(block_size=(2, 2), cell_size=(8, 8), vector_size=3)

    read = "DataSet/R-SCUT-FBP-"
    result = "/home/emre/Desktop/Feature Vectors/HOG/SCUT-FBP-"

    for i in range(index, last_index):
        path = read + str(i) + ".jpg"
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        featureVector = HOGDescriptor.compute(gray)

        fo = open(result + str(i) + ".txt", "w+")
        for j in range(len(featureVector)):
            fo.write(str(featureVector[j]) + "\n")
        fo.close()

        print(path, "islendi")

if __name__ == '__main__':

    start_time = time.time()

    extract_feature_all_images()

    end_time = time.time()

    print("Execution time: ", end_time - start_time)