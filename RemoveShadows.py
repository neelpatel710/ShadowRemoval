import numpy as np
import cv2
from tkinter import *
from tkinter import filedialog
import math

# image = []
height,width = 300,300
#37,64,150
image = cv2.imread('s6.jpg',-1)
# image = cv2.imread('Datasets/SBU-shadow/SBUTrain4KRecoveredSmall/ShadowImages/lssd401.jpg',-1)
image = cv2.resize(image,(300,300))

def openFile():
	global image,height,width
	filePath = filedialog.askopenfilename(title='Select your first image',filetypes=(('jpeg files','*.jpg'),('bmp files','*.bmp')))
	if filePath == "":
		print('No image is selected!')
		image = []
	else:
		image = cv2.imread(filePath)
		image = cv2.resize(image,(height,width))
		cv2.imshow('Original',image)

#Function to calculate i and j values such as [i-3,i-2,i-1,i,i+1,i+2,i+3],[j-3,j-2,j-1,j,j+1,j+2,j+3]
def getRange(i,j,height,width):
    lineThreshold = 3
    I,J = [],[]
    for p in range(0,lineThreshold+1):
        tmpp,tmps = int(i+p),int(i-p)
        if tmpp >= 0 and tmpp < height:
            I.append(tmpp)
        if tmps >= 0 and tmps < height and p != 0:
            I.append(tmps)
    for q in range(0,lineThreshold+1):
        tmpp,tmps = int(j+q),int(j-q)
        if tmpp >= 0 and tmpp < width:
            J.append(tmpp)
        if tmps >= 0 and tmps < width and q != 0:
            J.append(tmps)
    return I,J

def shadowRemovalFunction():
	global image,height,width
	if image == []:
		print("Select an Image")
	else:
		print("Performing Shadow Removal Technique..")
		labImage = cv2.cvtColor(image,cv2.COLOR_BGR2Lab)
		grayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		lightness,a,b = cv2.split(labImage)
		mean,std = np.mean(lightness),np.std(lightness)

		maskLabImage = np.copy(labImage)
		for i in range(maskLabImage.shape[0]):
		    for j in range(maskLabImage.shape[1]):
		        if lightness[i,j] > (mean-(std/3)):
		            #Consider to be non-shadow
		            grayImage[i,j] = 0
		                    
		binaryImage = grayImage.copy()
		for i in range(grayImage.shape[0]):
		    for j in range(grayImage.shape[1]):
		        if grayImage[i,j] != 0:
		            binaryImage[i,j] = 0
		        else:
		            binaryImage[i,j] = 255

		kernel = np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]],np.uint8)
		dilate = cv2.dilate(binaryImage,kernel,iterations=2)
		mask_lab = dilate.copy()
		# mask_lab = binaryImage.copy()
		cv2.imshow('Mask',mask_lab)
		cv2.imshow('Original',image)

		labelMatrix = np.zeros((image.shape[0],image.shape[1]))
		for i in range(mask_lab.shape[0]):
		    for j in range(mask_lab.shape[1]):
		        if mask_lab[i][j] == 0:
		            #shadow
		            labelMatrix[i][j] = 1111
		        else:
		            #non-shadow
		            labelMatrix[i][j] = 2222
		            
		imageMatrixB = image[:,:,0]
		imageMatrixG = image[:,:,1]
		imageMatrixR = image[:,:,2]
		imageMatrixA = labImage[:,:,1]

		indexOfShadowPixels = []    
		indexOfNonShadowPixels = []    
		# print(len(indexOfShadowPixels))
		# print(len(indexOfNonShadowPixels))

		for i in range(labelMatrix.shape[0]):
		    for j in range(labelMatrix.shape[1]):
		        if labelMatrix[i][j] == 1111:
		            shadowPixel = labImage[i,j,1]
		            rangeOfI, rangeOfJ = getRange(i,j,height,width)
		            nonShadowNearPixels = []
		            for fullRow in range(0,height):
		                for rj in rangeOfJ:
		                    if labelMatrix[fullRow][rj] == 2222:
		                        nonShadowNearPixels.append([fullRow,rj,labImage[fullRow,rj,1]])
		            for ri in rangeOfI:
		                for fullCol in range(0,width):
		                    if labelMatrix[ri][fullCol] == 2222:
		                        nonShadowNearPixels.append([ri,fullCol,labImage[ri,fullCol,1]])            
		            #Computing the Distance with Near Pixels
		            distance = []
		            for nsi in range(len(nonShadowNearPixels)):
		                tmp = (int(nonShadowNearPixels[nsi][2])-int(shadowPixel))**2
		#                 tmp = math.sqrt((int(nonShadowNearPixels[nsi][2][0])-int(shadowPixel[0]))**2 + (int(nonShadowNearPixels[nsi][2][1])-int(shadowPixel[1]))**2 + (int(nonShadowNearPixels[nsi][2][2])-int(shadowPixel[2]))**2)
		                if tmp >= 2:
		                    distance.append([tmp,nonShadowNearPixels[nsi][0],nonShadowNearPixels[nsi][1]])
		            distance.sort()
		            image[i,j] = image[distance[0][1],distance[0][2]]

		cv2.imshow('Final Image',image)
		print("Done")
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		mainWindow.destroy()

mainWindow = Tk()
mainWindow.geometry("300x200")
mainWindow.title("Computer Vision - Project")
innerTitle = Label(mainWindow,text="Shadow Removal",font=('arial',12,'bold')).place(x=50,y=10)
openBtn = Button(mainWindow,text="Select an image",relief=RAISED,command=openFile).place(x=50,y=50)
runBtn = Button(mainWindow,text="Remove Shadow",relief=RAISED,command=shadowRemovalFunction).place(x=50,y=90)
mainWindow.mainloop()