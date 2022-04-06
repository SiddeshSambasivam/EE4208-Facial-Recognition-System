
# In[1]:


import cv2
import glob
import os


# In[2]:


inputFolder = 'resized_dataset'

i=0

for img in glob.glob(inputFolder + "/*.png"):
    image = cv2.imread(img)
    imgResized = cv2.resize(image, (300, 300))
    cv2.imwrite("Resized Folder/image%.png" %i, imgResized)
    
    i +=1
    cv2.imshow('image', imgResized)
    cv2.waitKey(30)
    
cv2.destroyAllWindows()    


# In[ ]:




