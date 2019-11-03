# Python program to read image using OpenCV 
  
# importing OpenCV(cv2) module 
import cv2 
import numpy as np
import math
# Save image in set directory 
# Read RGB image 
img = np.zeros((225,225))
img = cv2.imread('Houses-225.bmp',cv2.IMREAD_GRAYSCALE)  
print(np.array(img))

W = np.random.randn(7, 7)
print("W ")
print(W)
#print(img)
print(img.shape)
slice1 = img[:,:]
print(slice1.shape)
print("before slice")
print(slice1[0:7,0:7])
#print(slice1[])
#f= open("guru88.txt","w+")
#for i in slice1:
#	f.write(str(i))
#f.close()


#print(slice1)

#filter1 = np.zeros((7,7))

filter1 = np.array([[1,1,2,2,2,1,1],
          [1,2,2,4,2,2,1],
          [2,2,4,8,4,2,2],
          [2,4,8,16,8,4,2],
          [2,2,4,8,4,2,2],
          [1,2,2,4,2,2,1],
          [1,1,2,2,2,1,1]])
print(filter1)


count = 0
calc = np.zeros((225,225))
for i in range(3,222):
    for j in range(3,222):
         slice2 = img[i-3:i+4,j-3:j+4]
         #print(filter1)
         #print(slice2)
         #print(slice2.shape)
         calc[i,j] = np.sum(filter1*slice2)/140
    
print(calc.shape)

calc = np.around(calc,decimals=1)
f0= open("guru1.txt","w+")
for i in calc:
	f0.write(str(i))
f0.close()


# Gradient finding
calcx = np.zeros((225,225))
calcy = np.zeros((225,225))
calcz = np.zeros((225,225))
calca = np.zeros((225,225))
#ca = np.negative(calc)
#print(ca.shape)
np.set_printoptions(suppress=True)
f= open("grad.txt","w+")
f2= open("grad2.txt","w+")
gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
slice3 = np.zeros((3,3))
for i in range(4,221):
    for j in range(4,221):
        slice3 = calc[i-1:i+2,j-1:j+2]
        #f2.write(str(slice3))
        calcx[i,j] = np.around((np.sum(gx*slice3))**2,decimals=1)
        calcy[i,j] = np.around((np.sum(gy*slice3))**2,decimals=1)
        calcz[i,j] = np.around(np.sqrt(calcx[i,j]+calcy[i,j]),decimals=1)
        if calcx[i,j] != 0:
            calca[i,j] = math.degrees(math.atan(calcy[i,j]/calcx[i,j]))
            f2.write(str(calcy[i,j]/calcx[i,j]))
            f2.write('\n')
        else:
             if calcy[i,j] != 0:
                 calca[i,j] = 90.0
        #calc[i,j] = np.around(np.sqrt((np.sum(gx*slice3))**2 + (np.sum(gy*slice3))**2),decimals=2)
#        print((np.sum(gx*slice3)**2))
    f.write(str(calcz[i,:]))
g = np.array([[-100,0,99],[-2000,0,20],[-10000,0,1]])
print(np.sum(g))
#f= open("grad.txt","w+")
#for i in calc:
#	f.write(str(i))
f.close()
f2.close()
print((-5)**2)
f3= open("gulu.txt","w+")
for i in calca:
	f3.write(str(i))
f3.close()

f3= open("gudu.txt","w+")
for i in calcx:
	f3.write(str(i))
f3.close()


calcb = np.zeros((225,225))
for i in range(4,221):
    for j in range(4,221):
        if calca[i,j] > -22.5 and calca[i,j] <= 22.5:
            calcb[i,j] = 0
        elif calca[i,j] > 22.5 and calca[i,j] <= 67.5:
            calcb[i,j] = 1
        elif calca[i,j] > 67.5 and calca[i,j] <= 112.5:
            calcb[i,j] = 2
        elif calca[i,j] > -67.5 and calca[i,j] <= -22.5:
            calca[i,j] = 3


f3= open("gudu.txt","w+")
for i in calcb:
	f3.write(str(i))
f3.close()

calcc = np.zeros((225,225))
for i in range(5,220):
    for j in range(5,220):
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

t1 = 60
t2 = 130
for k in range(1,40):
    for i in range(5,220):
        for j in range(5,220):
            if calcc[i,j] < t1:
                calcc[i,j] = 0
            elif calcc[i,j] > t2:
                calcc[i,j] = 255
            else:
                if max(calcc[i-1,j-1],calcc[i-1,j],calcc[i-1,j+1],calcc[i,j-1],calcc[i,j+1],calcc[i+1,j-1],calc[i+1,j],calcc[i+1,j+1])>t2 or min(abs(calca[i,j]-calca[i-1,j-1]),abs(calca[i,j]-calca[i-1,j]),abs(calca[i,j]-calca[i-1,j+1]),abs(calca[i,j]-calca[i,j-1]),abs(calca[i,j]-calca[i,j+1]),abs(calca[i,j]-calca[i+1,j-1]),abs(calca[i,j]-calca[i+1,j]),abs(calca[i,j]-calca[i+1,j+1])) <=45:
                    calcc[i,j] = 255


for i in range(5,220):
    for j in range(5,220):
        if calcc[i,j]<=t2 and calcc[i,j]>=t1:
            calcc[i,j] = 0

f3= open("gudu.txt","w+")
for i in calcc:
	f3.write(str(i))
f3.close()

cv2.imwrite('abcde.bmp', calcc)
#cv2.imshow("image", calcc)
#cv2.waitKey()

# Output img with window name as 'image' 
#cv2.imshow('image', img)  
  
# Maintain output window utill 
# user presses a key 
#cv2.waitKey(0)         
  
# Destroying present windows on screen 
#cv2.destroyAllWindows()  
