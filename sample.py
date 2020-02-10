import cv2
import numpy as np
import math
import scipy

def main():
    img = cv2.imread("image.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = 1/16 * np.mat([[1,2,1],[2,4,2],[1,2,1]])
    #kernel = 1/400 * np.mat ([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])
    kernel = calc_kernel(0.5)
    

    # sobel_x = np.mat([[2,2,4,2,2],[1,1,2,1,1],[0,0,0,0,0],[-1,-1,-2,-1,-1],[-2,-2,-4,-2,-2]])
    # sobel_y = np.mat([[2,1,0,-1,-2],[2,1,0,-1,-2],[4,2,0,-2,-4],[2,1,0,-1,-2],[2,1,0,-1,-2]])

    sobel_x = np.mat([[1,2,1],[0,0,0],[-1,-2,-1]])
    sobel_y = np.mat([[-1,0,1],[-2,0,2],[-1,0,1]])

    img2 = cross_correlation_2d(img,kernel)
    
    xsobel = cross_correlation_2d(img2,sobel_x)
    img3 = cross_correlation_2d(img2,sobel_y)
    grad, dirMat = gradient(xsobel, img3, 25)

    img2 = cv2.convertScaleAbs(img2)

    imgNone = non_max_suppression(grad, dirMat)
    
    

    grad = cv2.convertScaleAbs(grad)
    xsobel = cv2.convertScaleAbs(xsobel)
    img3 = cv2.convertScaleAbs(img3)
    imgNone = cv2.convertScaleAbs(imgNone)
    
    cv2.imshow("smooth",img2)
    cv2.imshow("xsobel",xsobel)
#     cv2.imshow("ysobelcv",cv2Canny)
    cv2.imshow("ysobel",img3)
    cv2.imshow("gradient",grad)
    cv2.imshow("non-max",imgNone)
    cv2.imshow("gradOr",dirMat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
        return new_img, ori
def calc_kernel(sigma):
    hsize = 2 * math.ceil(3 * sigma) + 1
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

