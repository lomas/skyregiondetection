import sys
import numpy as np
import cv2
import math

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

def check_building(img, H):
	h,w,c = img.shape
#	buildingflag = np.zeros((w,1))
	buildingflag = np.array([0 for i in range(w)])
	for x in range(w-1):
		if abs(H[x] - H[x+1]) > h / 4:
			if H[x] < H[x+1]:
				buildingflag[x] = 1
			else:
				buildingflag[x+1] = 1
	if sum(buildingflag) < 1:
		return H
	
	total_sample = sum(H)
	num_building = sum(H[buildingflag == 1])
	buildingsample = np.zeros((num_building,3))
	skysample = np.zeros((total_sample - num_building,3))
	skyidx = 0
	buildingidx = 0
	for x in range(w):
		if buildingflag[x] == 0:
			for y in range(H[x]):
				skysample[skyidx,0] = img[y,x,0]
				skysample[skyidx,1] = img[y,x,1]
				skysample[skyidx,2] = img[y,x,2]
				skyidx = skyidx + 1
		else:
			for y in range(H[x]):
				buildingsample[buildingidx,0] = img[y,x,0]
				buildingsample[buildingidx,1] = img[y,x,1]
				buildingsample[buildingidx,2] = img[y,x,2]
				buildingidx = buildingidx + 1
	skycov,skymean = cv2.calcCovarMatrix(skysample, cv2.COVAR_ROWS|cv2.COVAR_NORMAL)
	buildingcov,buildingmean = cv2.calcCovarMatrix(buildingsample, cv2.COVAR_ROWS|cv2.COVAR_NORMAL)
	ret,skycovInv = cv2.invert(skycov)
	ret,buildingcovInv = cv2.invert(buildingcov)
	for x in range(w):
		if buildingflag[x] == 1:
			continue
		idx = 0
		sample = np.zeros((H[x], 3))
		for y in range(H[x]):
			sample[idx,0] = img[y,x,0]
			sample[idx,1] = img[y,x,1]
			sample[idx,2] = img[y,x,2]
			idx = idx + 1
		num_sample = sample.shape[0]
		ss = sample - np.tile(skymean,(num_sample,1))
		dsky = np.dot(np.dot(ss , skycovInv),ss.T)
		ss = sample - np.tile(buildingmean,(num_sample,1))
		dbuilding = np.matrix(ss) * np.matrix(buildingcovInv) * np.matrix(ss).T
		dsky = np.diag(dsky)
		dbuilding = np.diag(dbuilding)
		num_sky = sum(dsky < dbuilding)
		if num_sky * 2 < num_sample:
			buildingflag[x] = 1
	for x in range(w):
		if buildingflag[x] == 1:
			H[x] = 0
	return(H)
	
def the_main(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	edgeimg = cv2.Sobel(gray, -1, 1,0)
	Jopt = 0
	Hopt = []
	for t in range(10,100,10):
		H = get_sky_boundary(edgeimg, t)
		Jm = calcJm(img, H)
		if Jopt < Jm:
			Jopt = Jm
			Hopt = H

	#Hopt = check_building(img, Hopt)
	return(Hopt)

if __name__ == "__main__":
	img = cv2.imread("E:\\tmp\\building.jpg")
	H = the_main(img)
	for x in range(img.shape[1]):
		y = H[x]
		pt = (x,y)
		cv2.circle(img, pt, 2, (1,0,0),2)
	cv2.imwrite("E:\\tmp\\sky_result.jpg", img)
