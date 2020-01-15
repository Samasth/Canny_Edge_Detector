import cv2 
import numpy as np
import math
import sys

img = cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE)  

print(img.shape)
h,w = img.shape
print("height=",h)
print('width=',w)

#Method to perform Gaussian smoothing
def gaussian_smoothing(img):

    # (7*7) gaussian filter for smoothing
    filter1 = np.array([[1,1,2,2,2,1,1],                            
                        [1,2,2,4,2,2,1],
                        [2,2,4,8,4,2,2],
                        [2,4,8,16,8,4,2],
                        [2,2,4,8,4,2,2],
                        [1,2,2,4,2,2,1],
                        [1,1,2,2,2,1,1]])

    calc = np.zeros((h,w))
    slice2 = np.zeros((7,7))
    
    # Performing cross convolution operation
    for i in range(3,h-3):                                          
        for j in range(3,w-3):
            slice2 = img[i-3:i+4,j-3:j+4]
            calc[i,j] = np.sum(filter1*slice2)/140
    calc=cv2.convertScaleAbs(calc)
    cv2.imshow("gauss", calc)
    cv2.imwrite('gaussian_smoothing.png', calc)
            #cv2.waitKey()
    return calc

# Method to perform Gradiant operations
def gradient (calc):
    
    calcx = np.zeros((h,w))
    calcy = np.zeros((h,w))
    calcz = np.zeros((h,w))
    calca = np.zeros((h,w))
    calcxo = np.zeros((h,w))
    calcyo = np.zeros((h,w))
    #np.set_printoptions(suppress=True)
    
    gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])                 # sobel operator for horizontal gradiant
    gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])                 # sobel operator for vertical gradiant
    slice3 = np.zeros((3,3))
    # Performing Gradiant operations
    for i in range(4,h-4):
        for j in range(4,w-4):
            slice3 = calc[i-1:i+2,j-1:j+2]
            # Computing horizontal and vertical gradiants
            calcxo[i,j] = (np.sum(gx*slice3))
            calcyo[i,j] = (np.sum(gy*slice3))

    x_max = np.amax(calcxo)
    y_max = np.amax(calcyo)

    # Performing normalisation to keep the pixel values in the range 0-255
    np.divide(calcxo, x_max)
    np.divide(calcyo, y_max)
    np.multiply(calcxo, 255)
    np.multiply(calcyo, 255)

    for i in range(4,h-4):
        for j in range(4,w-4):
            # Calculating Gradiant Magnitude
            slice3 = calc[i-1:i+2,j-1:j+2]
            calcx[i,j] = (np.sum(gx*slice3))**2
            calcy[i,j] = (np.sum(gy*slice3))**2
            calcz[i,j] = np.sqrt(calcx[i,j]+calcy[i,j])
            #Calculating Gradiant Angle
            if calcxo[i,j] != 0:
                calca[i,j] = math.degrees(math.atan(calcyo[i,j]/calcxo[i,j]))
            else:
                if calcyo[i,j] != 0:
                    calca[i,j] = 90.0
                else:
                    calca[i,j] = 0
    # Calculating absolute values of horizontal and vertical gradiants
    calcxo = np.absolute(calcxo)
    calcyo = np.absolute(calcyo)

    # Displaying Horizontal and Vertical Gradiants image
    calcxo = cv2.convertScaleAbs(calcxo)
    cv2.imshow("Vertical gradiant", calcxo) 
    cv2.imwrite('vertical_gradiant.png', calcxo)
    calcyo = cv2.convertScaleAbs(calcyo)
    cv2.imshow("Horizontal gradiant", calcyo)                
    cv2.imwrite('horizontal_gradiant.png', calcyo)

    # Displaying Gradiant magnitude image
    calcz = cv2.convertScaleAbs(calcz)
    cv2.imshow("gradiant Magnitude", calcz)
    cv2.imwrite('gradiant_magnitude.png', calcz)
    
    return calca,calcz


# Method to perform Non-Maxima suppression
def non_maxima_supression(calca,calcz):
    
    calcb = np.zeros((h-4,w-4))
    for i in range(4,h-4):
        for j in range(4,w-4):
            # Assigning gradiant angles to sectors in the range 0-3
            if calca[i,j] > -22.5 and calca[i,j] <= 22.5:                           
                calcb[i,j] = 0
            elif calca[i,j] > 22.5 and calca[i,j] <= 67.5:
                calcb[i,j] = 1
            elif calca[i,j] > 67.5 and calca[i,j] <= 112.5:
                calcb[i,j] = 2
            elif calca[i,j] > -67.5 and calca[i,j] <= -22.5:
                calca[i,j] = 3

    # Checking if the Gradiant magnitude of center value is greater than its adjacent values along the gradiant angle
    calcc = np.zeros((h,w))
    for i in range(5,h-5):
        for j in range(5,w-5):
            if calcb[i,j] == 0:
                if calcz[i,j]>calcz[i,j-1] and calcz[i,j]>calcz[i,j+1]:             
                    calcc[i,j] = calcz[i,j]
                else:
                    calcc[i,j] = 0
            elif calcb[i,j] == 1:
                if calcz[i,j]>calcz[i-1,j+1] and calcz[i,j]>calcz[i+1,j-1]:
                    calcc[i,j] = calcz[i,j]
                else:
                    calcc[i,j] = 0
            elif calcb[i,j] == 2:
                if calcz[i,j]>calcz[i-1,j] and calcz[i,j]>calcz[i+1,j]:
                    calcc[i,j] = calcz[i,j]
                else:
                    calcc[i,j] = 0
            elif calcb[i,j] == 3:
                if calcz[i,j]>calcz[i-1,j-1] and calcz[i,j]>calcz[i+1,j+1]:
                    calcc[i,j] = calcz[i,j]
                else:
                    calcc[i,j] = 0

    calcc = cv2.convertScaleAbs(calcc)
    cv2.imshow("After non-maxima suppression", calcc)
    cv2.imwrite('non_maxima_supp.png', calcc)
    return calcc,calca

# Method to perform double thresholding
def double_thresholding (calcc,calca):
    # selecting two threshold values    
    t1 = 60
    t2 = 100
    # dividing the image into three parts based on threshold values t1 and t2
    for i in range(5,h-5):
        for j in range(5,w-5):
            if calcc[i,j] < t1:
                calcc[i,j] = 0
            elif calcc[i,j] > t2:
                calcc[i,j] = 255
            else:
                if max(calcc[i-1,j-1],calcc[i-1,j],calcc[i-1,j+1],calcc[i,j-1],calcc[i,j+1],calcc[i+1,j-1],calcc[i+1,j],calcc[i+1,j+1])>t2 or min(abs(calca[i,j]-calca[i-1,j-1]),abs(calca[i,j]-calca[i-1,j]),abs(calca[i,j]-calca[i-1,j+1]),abs(calca[i,j]-calca[i,j-1]),abs(calca[i,j]-calca[i,j+1]),abs(calca[i,j]-calca[i+1,j-1]),abs(calca[i,j]-calca[i+1,j]),abs(calca[i,j]-calca[i+1,j+1])) <=45:
                    calcc[i,j] = 255

    # second pass
    for i in range(5,h-5):
        for j in range(5,w-5):
            if calcc[i,j] <= t2 and calcc[i,j] >= t1:
                calcc[i,j] = 0

    cv2.imwrite('thresholding.png', calcc)
    cv2.imshow("Final image after thresholding", calcc)
    cv2.waitKey()
    
    
calcf = gaussian_smoothing(img)
calcaf,calczf = gradient(calcf)
calccf,calcaf = non_maxima_supression(calcaf,calczf)
double_thresholding(calccf,calcaf)



