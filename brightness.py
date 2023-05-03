#Increase or decrease brightness

import cv2 
import numpy as np

def change_brightness(img, brighten=0, darken=0):

    #GEt the hsv colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #print(hsv.dtype)
    h, s, v = cv2.split(hsv)


    darken_true = -1*(darken)
    #Select a value between the lower and upper bound
    brightness_factor = np.random.uniform(darken_true,brighten)
    print(brightness_factor)
    value = brightness_factor*255

    #Overflowing to 0 when sum is higher than 255, add condition. 
    #Same occurs when you go darken below 0, it will overflow to 255 (white)
    if brightness_factor>0:
        mask = (255 - hsv[:,:,2]) < 255*brightness_factor

        hsv[:,:,2] = np.where((255-hsv[:,:,2])< value, 255, hsv[:,:,2]+value)
    elif brightness_factor<0:
        hsv[:,:,2] = np.where((hsv[:,:,2])< abs(value), 0, hsv[:,:,2]+value)



    #np.savetxt("myarray.txt",hsv[:,:,2])
    #print(np.amin(hsv[:,:,2]))

    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return result


if __name__ == "__main__":
    image = cv2.imread("data-augmentation/13.jpg")
    image_v = change_brightness(image, 0, 1)
    # display the original and the Translated image

    cv2.imshow('Original image', image)
    cv2.waitKey(0)

    cv2.imshow('Brightness image', image_v)
    cv2.waitKey(0)
    cv2.imwrite("test_brightness.jpg", image_v)

    

