import cv2

def image_resize(image, width = None, height = None, inter = cv2.INTER_CUBIC):

    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

if __name__ == '__main__':

    for i in range(1, 501):
        input = cv2.imread("Data_Collection/SCUT-FBP-"+str(i)+".jpg", cv2.IMREAD_COLOR)
        output = image_resize(input, height=640)
        cv2.imwrite("DataSet/R-SCUT-FBP-"+str(i)+".jpg", output)
        print(i, "islendi")