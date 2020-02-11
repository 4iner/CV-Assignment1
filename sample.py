import cv2
import numpy as np
import math
import scipy
import matplotlib.pyplot as plt

im = []
tit = []

def myEdgeFilter(img0, sigma):
    hsize = 2 * math.ceil ( 3 * sigma) + 1
    kernel = calc_kernel(hsize, sigma)

    print("Convolving image with gauss of size %d..." % (hsize))
    img_smooth = convolve_2d(img0, kernel)
    im.append(img_smooth)
    tit.append("Smoothed")
    sobel_x = np.mat([[1,2,1],[0,0,0],[-1,-2,-1]])
    sobel_y = np.mat([[-1,0,1],[-2,0,2],[-1,0,1]])
    print("Convolving image with x sobel...")
    img_x = convolve_2d(img_smooth, sobel_x)
    im.append(img_x)
    tit.append("X Sobel")
    print("Convolving image with y sobel...")
    img_y = convolve_2d(img_smooth, sobel_y)
    im.append(img_y)
    tit.append("Y Sobel")

    print("Finding gradient dir and magnitude...")
    gradMag, gradDir = gradient(img_x, img_y, 50)
    im.append(gradMag)
    tit.append("Gradient")
    print("Performing non-maxima suppression...")
    img_non_max = non_max_suppression(gradMag, gradDir)
    im.append(img_non_max)
    tit.append("Non-maxima")
    thresh_calc(img_non_max,100,200)
    return img_non_max



def main():
    img_cat = cv2.imread("cat.png",cv2.IMREAD_GRAYSCALE)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_dog = cv2.imread("dog.png",cv2.IMREAD_GRAYSCALE)
  

    khp = np.mat([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    # canny = myEdgeFilter(img, 0.25) 
    # im.append(canny)
    # tit.append("Canny")
    gauss_3x3 = calc_kernel(3, 0.25)

    fil = high_pass(img_cat, calc_kernel(51,5))
    im.append(fil)
    tit.append("high pass")

    fil2 = low_pass(img_dog, gauss_3x3)
    im.append(fil2)
    tit.append("low pass")

    hybrid = fil + fil2
    im.append(hybrid)
    tit.append("hybrid")
    # cv2.imshow("hipghas",hp)
    # cv2.imshow("im2",img2)
    # cv2.imshow("image",img)

    # plt.show()
    show_images(im,titles=tit)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def high_pass(img, kernel):
    smooth_img = convolve_2d(img, kernel)
    fil = img - smooth_img
    return fil
    
def low_pass(img, kernel):
    return convolve_2d(img, kernel)


def show_images(images, cols = 1, titles = None):
    #https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
    """Display a list of images in a single figure with matplotlib.
   
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
   
    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).
   
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        # if image.ndim == 2:
        #     plt.gray()
        plt.imshow(image, cmap='gray')
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images + 1000)
    plt.show()
def cross_correlation_2d(image, kernel):
    m, n = kernel.shape
    if (m == n):
        y,x = image.shape
        offset = math.floor(m/2)
        padded_img = np.zeros((y + offset * 2, x + offset * 2))
        padded_img[offset:-offset, offset:-offset] = image
        result = np.zeros(image.shape)
        ke = np.array(kernel) # array multiplication is faster
        im = np.array(padded_img) 
        for i in range(y):
            for j in range(x):
                res = np.sum(im[i:i+m, j:j+m] * ke)
                if (res < 0):
                     res = 0
                if (res > 255):
                    res = 255
                result[i][j] = res
        return np.mat(result)
    return [[]]


def convolve_2d(image, kernel):
    kernel = rotate_matrix_180(kernel)
    return cross_correlation_2d(image, kernel)
    
def rotate_matrix_180(kernel):
    new_ker = kernel.copy()
    for i in range(kernel.shape[0]): 
        h = 0
        k = kernel.shape[0]-1
        while h < k: 
            t = kernel[h,i] 
            new_ker[h,i] = kernel[k,i] 
            new_ker[k,i] = t 
            h += 1
            k -= 1
    return new_ker
def thresh_calc(img, t1, t2):
    y, x = img.shape
    for i in range(y):
        for j in range(x):
            if img[i,j] < t1:
                img[i,j] = 0
            elif img[i,j] >= t1 and img[i,j] < t2:
                img[i,j] = t1
            else:
                img[i,j] = t2

def non_max_suppression(grad, dirmat):
    y, x = grad.shape
    new_grad = np.zeros((y,x), dtype=np.int32)
    angles = np.array(dirmat)
    angles[angles < 0] += 180

    for i in range(1, y - 1):
        for j in range(1, x - 1):
            try:
                c = 255
                k = 255
                if 0 <= angles[i,j] < 22.5 or 157.5 <= angles[i,j] <= 180:
                    c = grad[i, j+1]
                    k = grad[i, j-1]
                elif 22.5 <= angles[i,j] < 67.5:
                    c = grad[i+1, j-1]
                    k = grad[i-1, j+1]
                elif 67.5 <= angles[i,j] < 112.5:
                    c = grad[i+1, j]
                    k = grad[i-1, j]
                elif 112.5 <= angles[i,j] < 157.5:
                    c = grad[i-1, j-1]
                    k = grad[i+1, j+1]

                if grad[i,j] >= c and grad[i,j] >= k:
                    new_grad[i,j] = grad[i,j]
                else:
                    new_grad[i,j] = 0

            except IndexError as e:
                pass
    
    return new_grad
def gradient(imagex, imagey, threshold):
    if(imagex.shape == imagey.shape):
        new_img = np.zeros(imagex.shape)
        y,x = imagex.shape
        for i in range(y):
            for j in range(x):
                sum = math.sqrt(math.pow(imagex[i,j],2) + math.pow(imagey[i,j],2))
                if(sum > threshold):
                    new_img[i,j] = sum
                else:
                    new_img[i,j] = 0
        ori = np.degrees(np.arctan2(imagex,imagey))
        new_img = new_img * (255 / new_img.max())
        return new_img, ori
def calc_kernel(hsize, sigma):
    kernel = np.zeros((hsize,hsize))
    offset = math.floor(hsize/2)
    sum = 0
    for h in range(-1 * offset, offset + 1):
        for k in range(-1 * offset, offset + 1):
            kernel[h + offset,k + offset] = calc_gauss(sigma, h, k)
            sum += kernel[h,k]

    kernel = kernel / np.sum(kernel);
    return kernel

def calc_gauss(sigma, u, v):
    e_exp = -1 * ((math.pow(u, 2) + math.pow(v, 2))/(2 *math.pow(sigma,2)))
    val = (1 / (2 * math.pi * math.pow(sigma, 2))) * math.pow(math.e,e_exp)
    return val
    #print("Result: ",result[i][j])
# def my_convolution(image,kernel):
#     m, n = kernel.shape
#     if m == n:
#         y, x = image.shape
#         offset = math.floor(m/2)
#         padded_img = np.zeros((y+offset*2,x+offset*2))
#         padded_img[offset:-1 * offset,offset:-1 * offset] = image
#         result = np.zeros(image.shape)
#         for i in range(y):
#             for j in range(x):
#                 result[i][j] = 0
#                 for h in range(-1 * offset, offset + 1):
#                     for k in range(-1 * offset, offset + 1):
#                         result[i][j] += math.floor( kernel[h,k] * padded_img[i-h,j-k] )
#                 #print("Result: ",result[i][j])
#         print(result)
#         return result
#     return [[]]


if __name__ == "__main__":
    main()

