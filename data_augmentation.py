#Check the correct transformation 
import cv2
import numpy as np 
import xml.etree.ElementTree as ET
from rotation import rotate_im
import shutil
import argparse

parser = argparse.ArgumentParser()

#Example of arguments: 
parser.add_argument("-tx", "--translationx", help="Translation in the X direction")
parser.add_argument("-ty", "--translationy", help="Translation in the Y direction")
parser.add_argument("-p", "--path", help="Images path")

args = parser.parse_args()

#Read the image
image = cv2.imread("/home/mario3/arinapin/python/git/deeplearning-apps/data-augmentation/13.jpg")

height, width = image.shape[:2]
#tx, ty = int(args.translationx), int(args.translationy)


#Translation
tx,ty = -100, -100 
translation_matrix = np.array([
    [1,0,tx],
    [0,1,ty]
], dtype=np.float32)

# Get the amount of padding needed
if tx >= 0:
    right = 0
    left = tx
    
else:
    right = abs(tx)
    left = 0
    

if ty >= 0: 
    top = ty
    bottom = 0
else: 
    top = 0
    bottom = abs(ty)
    
#top, bottom, left, right = abs(ty), abs(ty), abs(tx), abs(tx)

translated_image = cv2.warpAffine(src=image, M=translation_matrix, dsize=(width, height))
print(translated_image.shape)
padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REPLICATE)


#crop the height and width
if tx >= 0:
    if ty >=0:
        translated_image2 = padded_image[0:height, 0:width]
    else:
        translated_image2 = padded_image[abs(ty):(height+abs(ty)), 0:width]
elif tx < 0:
    if ty >=0:
        translated_image2 = padded_image[0:height, abs(tx):width+abs(tx)]
    else:
        translated_image2 = padded_image[abs(ty):(height+abs(ty)), abs(tx):width+abs(tx)] 


#Rotation
degree = 90
rotated_im=rotate_im(image=image, angle=degree)

#copy xml file
src_anotations="/home/mario3/arinapin/python/git/deeplearning-apps/data-augmentation/13.xml"
dest_directory="/home/mario3/arinapin/python/git/deeplearning-apps/data-augmentation/"
dest_file = dest_directory+"result.xml"
shutil.copy2(src_anotations, dest_file)

#use 
def read_content_andtranslation(xml_file: str, dest_file, tx, ty):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    #load the destination of translation
    tree2 = ET.parse(dest_file)
    root2 = tree2.getroot()
    
    list_with_all_boxes = []
    list_with_result_boxes = []
    #Get image size for bbox translation 
    size_element = root.find('size')

    # Get the width and height values
    width = int(size_element.find('width').text)
    height = int(size_element.find('height').text)

    for boxes in root.iter('object'):

        #Get the file name
        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None
        clase = str(boxes.find("name").text) 
        ymin = int(boxes.find("bndbox/ymin").text)
        ymin2 = ymin+ty
        #Saturate the values for area calculations
        if ymin2 < 0:
            ymin2 = 0
        if ymin2 > height:
            ymin2 = height
        xmin = int(boxes.find("bndbox/xmin").text)
        xmin2 = xmin+tx
        if xmin2 < 0: 
            xmin2 = 0
        if xmin2 > width:
            xmin2 = width
        
        ymax = int(boxes.find("bndbox/ymax").text)
        ymax2 = ymax+ty
        if ymax2 > height:
            ymax2 = height
        if ymax2 <0:
            ymax2 = 0
        xmax = int(boxes.find("bndbox/xmax").text)
        xmax2 = xmax+tx
        if xmax2 > width: 
            xmax2 = width
        if xmax2 <0: 
            xmax2 = 0

        orig_area = (xmax-xmin)*(ymax-ymin)
        
        #Make the bbox 0 
        new_area = (xmax2-xmin2)*(ymax2-ymin2)

        #Delete bbox when the new area is 60% less than the original
        reduction = 0.6
        if new_area < orig_area*(1-reduction):
            xmin2=ymin2=xmax2=ymax2=-1




        #Remove the bbox where 60% or more has been left out of bounds
        # if ymin2 < 0 or xmin2 < 0: 
        #     ymin2 = xmin2 = -1
        #     xmax2 = ymax2 = -1 
        # elif ymax2 > height:
        #     ymin2 = xmin2 = -1
        #     xmax2 = ymax2 = -1 
        # elif xmax2 > width: 
        #     ymin2 = xmin2 = -1
        #     xmax2 = ymax2 = -1 

        list_with_single_boxes = [clase, xmin, ymin, xmax, ymax]
        list_with_result_box = [xmin2, ymin2, xmax2, ymax2]

        
        list_with_all_boxes.append(list_with_single_boxes)
        list_with_result_boxes.append(list_with_result_box)

    #We need to delete bndbox leaving the original image size, ...
    
    #Modify with the resulting transformation
    for boxes, value in zip(root2.iter('object'), list_with_result_boxes):

        ymin = boxes.find("bndbox/ymin")
        xmin = boxes.find("bndbox/xmin")
        ymax = boxes.find("bndbox/ymax")
        xmax = boxes.find("bndbox/xmax")

        ymin.text = str(value[1])
        xmin.text = str(value[0])
        ymax.text = str(value[3])
        xmax.text = str(value[2])

    #delete bbox with -1, because it is out of bounds. 
    for obj in root2.findall('object'):
        # Check if a condition is true
        if obj.find('bndbox/ymin').text == '-1':
            # If the condition is true, remove the object element
            root2.remove(obj)

    #write the result to the file 
    tree2.write(dest_file)


    return filename, list_with_all_boxes, list_with_result_boxes

name, boxes, result_boxes = read_content_andtranslation(src_anotations, dest_file, tx, ty)

print(boxes)
print(result_boxes)

#Draw rectangle
color = (0, 0, 255)  # Rojo en formato BGR
grosor = 2

for box in boxes:
    cv2.rectangle(image, (box[1], box[2]), (box[3], box[4]), color, grosor)

for box in result_boxes:
    cv2.rectangle(translated_image, (box[0], box[1]), (box[2], box[3]), color, grosor)

# display the original and the Translated image
cv2.imshow('Original image', image)
cv2.waitKey(0)
cv2.imshow('Translated image', translated_image)
cv2.imshow('Translated 2 reflected', translated_image2)
cv2.waitKey(0)


#Check shape of new translated image
print("Padded_image:", padded_image.shape)


print(translated_image2[:,0:100,1])
print(translated_image2.shape)

#Apply transformation to each one. 
cv2.imwrite("result4.jpg", translated_image2)
cv2.imwrite("result-wrap.jpg", translated_image)
