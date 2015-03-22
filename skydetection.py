import os 
import sys
import numpy as np
import cv2


def get_sky_boundary(edgeimg, thresh):
	h,w = edgeimg.shape
	H = np.array([-1 for k in range(w)])
	for x in range(w):
		for y in range(h):
			if edgeimg[y,x] > thresh:
				break
		H[x] = y
	return(H)

def calcJm(img, H):
	h,w,c = img.shape
	skysample = np.zeros((sum(H),3))
	gndsample = np.zeros((w*h - sum(H),3))
	ks = 0
	kg = 0 
	for x in range(w):
		for y in range(H[x]):
			skysample[ks,0] = img[y,x,0]
			skysample[ks,1] = img[y,x,1]
			skysample[ks,2] = img[y,x,2]
			ks = ks + 1
		for y in range(h - H[x]):
			gndsample[kg,0] = img[y + H[x], x, 0]
			gndsample[kg,1] = img[y + H[x], x, 1]
			gndsample[kg,2] = img[y + H[x], x, 2]
			kg = kg + 1
	skycov,skymean = cv2.calcCovarMatrix(skysample, cv2.COVAR_ROWS|cv2.COVAR_NORMAL)
	gndcov,gndmean = cv2.calcCovarMatrix(gndsample, cv2.COVAR_ROWS|cv2.COVAR_NORMAL)
	skyret,skyeigval,skyeigvec = cv2.eigen(skycov,1)
	gndret,gndeigval,gndeigvec = cv2.eigen(gndcov,1)
	skyeigvalsum = sum(skyeigval)
	gndeigvalsum = sum(gndeigval)
	skydet = cv2.determinant(skycov)
	gnddet = cv2.determinant(gndcov)
	Jm = 1.0 / (skydet + gnddet + skyeigvalsum*skyeigvalsum + gndeigvalsum * gndeigvalsum)
	return(Jm)
	

def the_main(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	edgeimg = cv2.Sobel(gray, -1, 1,0)
	Jopt = 0
	Hopt = []
