import cv2
import numpy as np
import math

def main():
    img = cv2.imread("image.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = 1/16 * np.mat([[1,2,1],[2,4,2],[1,2,1]])
    #kernel = 1/400 * np.mat ([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])
    kernel = calc_kernel(1.5)
    


    sobel = np.mat([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel2 = np.mat([[1,2,1],[0,0,0],[-1,-2,-1]])
    img2 = my_convolution(img,kernel)
    xsobel = my_convolution(img2,sobel)
    img3 = my_convolution(img2,sobel2)
    
    xsobel = cv2.convertScaleAbs(xsobel)
    img2 = cv2.convertScaleAbs(img2)
    img3 = cv2.convertScaleAbs(img3)
    cv2.imshow("smooth",img2)
    cv2.imshow("xsobel",xsobel)
    cv2.imshow("cvxsobel",xsobelcv)
    cv2.imshow("cvysobel",ysobelcv)
    cv2.imshow("ysobel",img3)
    cv2.imshow("cvsmooth",smoothsv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def my_convolution(image, kernel):
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

