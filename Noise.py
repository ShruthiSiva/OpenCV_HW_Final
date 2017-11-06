import cv2
import numpy as np 
import os
import glob


def Add_gaussian_Noise(img, mean, sigma):
	noisy_image=img.copy()


	cv2.randn(noisy_image, mean, sigma)
	cv2.add(img, noisy_image, noisy_image)
	
	return noisy_image

def Add_salt_pepper_Noise(img, pa, pb):
	noisy_image=img.copy()
	amt1=int(img.size*pa)
	amt2=int(img.size*pb)

	for i in range(amt1):
		noisy_image[np.random.randint(0, img.shape[0]-1), np.random.randint(0, img.shape[1]-1)]=0
	for i in range(amt2):
		noisy_image[np.random.randint(0, img.shape[0]-1), np.random.randint(0, img.shape[1]-1)]=255

	return noisy_image

mean=[0,5,10,20]
sigma=[0,20,50,100]
pa=[0.01, 0.03, 0.05, 0.4]
pb=[0.01, 0.03, 0.05, 0.4]
for a in pa:
	for b in pb:
		for img in glob.glob("Test_images/*.png"):
			image=cv2.imread(img,0)
			noisy_image=Add_salt_pepper_Noise(image, a, b)
			cv2.imwrite("{}_Pa{}_Pb{}.jpg".format(img,a,b), noisy_image)
while False:
	for m in mean:
		for s in sigma:
			for img in glob.glob("Test_images/*.png"):

				image=cv2.imread(img,0)

				noisy_image=Add_gaussian_Noise(image, m,s)


				cv2.imwrite("{}_Mean{}_Sigma{}.jpg".format(img,m,s), noisy_image)


cv2.waitKey(0)
cv2.destroyAllWindows()