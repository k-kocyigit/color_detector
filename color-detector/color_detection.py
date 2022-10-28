import numpy as np
import cv2

kamera=cv2.VideoCapture(0)

while True:
    _, imageFrame=kamera.read()
    hsvFrame=cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
    
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
    
    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
    
    black_lower = np.array([0, 0, 0], np.uint8)
    black_upper = np.array([90, 90, 90], np.uint8)
    black_mask = cv2.inRange(hsvFrame, black_lower, black_upper)
    
    kernal=np.ones((5,5), "uint8")
    
    red_mask=cv2.dilate(red_mask,kernal)
    res_red=cv2.bitwise_and(imageFrame, imageFrame,mask=red_mask)
    
    blue_mask = cv2.dilate(blue_mask, kernal)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame,mask = blue_mask)
    
    black_mask = cv2.dilate(black_mask, kernal)
    res_black = cv2.bitwise_and(imageFrame, imageFrame,mask = black_mask)
                                
    #for red
    contours, hierarchy = cv2.findContours(red_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                       (x + w, y + h), 
                                       (0, 0, 255), 2)
              
            cv2.putText(imageFrame, "Kirmizi", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255))
    #for blue
    contours, hierarchy = cv2.findContours(blue_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (255, 0, 0), 2)
              
            cv2.putText(imageFrame, "Mavi", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 0, 0))
    #for black
    contours, hierarchy = cv2.findContours(black_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
      
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                       (x + w, y + h),
                                       (0, 0, 0), 2)
              
            cv2.putText(imageFrame, "Siyah", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (0, 0, 0))
    cv2.imshow("Kirmizi Mavi Siyah Renk Dedektoru", imageFrame),
    if cv2.waitKey(1) & 0xFF == ord('q'):
        kamera.release()
        cv2.destroyAllWindows()
        break
                                           