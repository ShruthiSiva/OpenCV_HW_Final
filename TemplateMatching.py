import numpy as np
import cv2

def TemplateMatching(src, temp, stepsize): # src: source image, temp: template image, stepsize: the step size for sliding the template
    mean_t = 0
    var_t = 0
    location = [0, 0]
    # Calculate the mean and variance of template pixel values
    # ------------------ Put your code below ------------------ 

    m,s=cv2.meanStdDev(temp)
    v=[(x**2) for x in s]
    mean=list(np.uint64(m))
    mean[0]
    variance=list(np.uint64(v))
    variance=variance[0]
    
    
                
    max_corr = 7
    # Slide window in source image and find the maximum correlation
    for i in np.arange(0, src.shape[0] - temp.shape[0], stepsize):
        for j in np.arange(0, src.shape[1] - temp.shape[1], stepsize):
            mean_s = 0
            var_s = 0
            corr = 0

            # Calculate the mean and variance of source image pixel values inside window
            # ------------------ Put your code below ------------------ 
            
            imgtoview=src[j:j+temp.shape[1], i:i+temp.shape[0]]
            mean_s,std_s=cv2.meanStdDev(imgtoview)
            
            mean_s=list(mean_s)
            std_s=list(std_s)
            var_s=[(x**2) for x in std_s]
            mean_s=list(np.uint64(mean_s))
            mean_s=mean_s[0]
            var_s=list(np.uint64(var_s))
            var_s=var_s[0]


            # Calculate normalized correlation coefficient (NCC) between source and template
            # ------------------ Put your code below ------------------ 
            NCC=1/temp.size
            for p in np.arange(0, src.shape[0] - temp.shape[0], stepsize):
                for q in np.arange(0, src.shape[1] - temp.shape[1], stepsize):
                    for a in np.arange(0, temp.shape[0], stepsize):
                        for b in np.arange(0, temp.shape[1], stepsize):
                            NCC=NCC+((src[p+a,q+b]-mean_s)*(temp[a,b]-mean))
            
            
            NCC=NCC/(var_s*variance)
            corr=NCC
            

            if corr > max_corr and corr!=np.inf:
                max_corr = corr
                
                location = [i, j]
            print("NCC: {}, MAX_CORR: {}, (i,j)={}".format(NCC, max_corr, (i,j)))
    return location

# load source and template images
source_img = cv2.imread('source_img.jpg',0) # read image in grayscale
#source_img=cv2.resize(source_img, (640, 480))
temp = cv2.imread('template_img.jpg',0) # read image in grayscale
#temp=cv2.resize(temp, (124, 161))
location = TemplateMatching(source_img, temp, 20)

print(location)
match_img = cv2.cvtColor(source_img, cv2.COLOR_GRAY2RGB)

# Draw a red rectangle on match_img to show the template matching result
# ------------------ Put your code below ------------------ 
src=cv2.rectangle(source_img, (160,380), (401, 380+138), (0,255,0),3)
# Save the template matching result image (match_img)
# ------------------ Put your code below ------------------ 


# Display the template image and the matching result
cv2.namedWindow('TemplateImage', cv2.WINDOW_NORMAL)
cv2.namedWindow('MyTemplateMatching', cv2.WINDOW_NORMAL)
cv2.imshow('TemplateImage', temp)
cv2.imshow('MyTemplateMatching', src)
cv2.waitKey(0)
cv2.destroyAllWindows()