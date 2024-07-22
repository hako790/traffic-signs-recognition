import numpy as np
import cv2
import pickle



def detect_traffic_signs(video):
    # شفرة البرنامج الأول
   
   # قراءة الفيديو
   cap = cv2.VideoCapture('C:/Users/Hako/Desktop/traffic signs/videos/IMG_5247.MOV')
   roi = [(400, 300), (1000,1200)]
   var=0
   while True:
       # إيجاد الإطار الحالي
       ret, frame = cap.read()
       
       x1, y1 = roi[0]
       x2, y2 = roi[1]
       frame = frame[y1:y2, x1:x2]
       
       # تحويل الإطار إلى صورة HSV
       hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
       
       # تعريف نطاق اللون الأحمر في HSV
       lower_red = np.array([0,50,50])
       upper_red = np.array([10,255,255])
       mask1 = cv2.inRange(hsv, lower_red, upper_red)
       
       lower_red = np.array([170,50,50])
       upper_red = np.array([180,255,255])
       mask2 = cv2.inRange(hsv, lower_red, upper_red)
       
      # تعريف نطاق اللون الأزرق في HSV
       lower_blue = np.array([100,50,50])
       upper_blue = np.array([130,255,255])
       mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

      
   # إضافة القناعين معاً
       mask0 = mask1 + mask2   
       mask = mask1 + mask2 + mask_blue
       
       # apply blur and Hough transform to detect circles
       blurred = cv2.GaussianBlur(mask, (7, 7), 0)
       circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 200, param1=50, param2=30, minRadius=25, maxRadius=50)
     
    
       # draw circles on the original frame
       if circles is not None:
           circles = np.round(circles[0, :]).astype("int")
           for (x, y, r) in circles:
               x1 = x - r-10
               y1 = y - r-10
               x2 = x + r+10
               y2 = y + r+10
               cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
               print("Found circles")
               cropped_frame = frame[y1:y2, x1:x2]
               classify_traffic_signs(cropped_frame)
               
             
               

       
       if cv2.waitKey(1) == ord('q'):
           break
       # تطبيق معالجة الضوضاء
       kernel = np.ones((5,5),np.uint8)
       opening = cv2.morphologyEx(mask0, cv2.MORPH_OPEN, kernel)
       
       # العثور على المناطق المغلقة
       contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
       
       for contour in contours:
         approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
         area = cv2.contourArea(contour)
         x, y, w, h = cv2.boundingRect(contour)
         w=w+30
         h=h+30
         x=x-10
         y=y-10
         if len(approx) == 3 and area > 200:
               # رسم المثلث
               
               
               cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
               print("Found triangle")
               cropped_frame = frame[y:y+h, x:x+w]
               if  var==3 or var==0:
                    var=0
                    classify_traffic_signs(cropped_frame)
                    var=var+1
               
               else:
                var=var+1
               classify_traffic_signs(cropped_frame)
       # عرض الفيديو
       cv2.imshow('Video',frame)
       
       
       # انتظار الضغط على مفتاح لإنهاء البرنامج
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

   # إغلاق الفيديو والنافذة
   cap.release()
   cv2.destroyAllWindows()
    
   locations =  cropped_frame
   return locations

def classify_traffic_signs(cropped_frame):
    # شفرة البرنامج الثاني
    threshold = 0.75         # PROBABLITY THRESHOLD
    font = cv2.FONT_HERSHEY_SIMPLEX
    ##############################################
     
    # SETUP THE VIDEO CAMERA

    imgOrignal = (cropped_frame )



    # IMPORT THE TRANNIED MODEL

    pickle_in=open("C:/Users/Hako/Desktop/traffic signs recognition/model_trained.p","rb")  ## rb = READ BYTE
    model=pickle.load(pickle_in)
     
    def grayscale(img):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        
        return img
    def equalize(img):
        img =cv2.equalizeHist(img)
        return img
    def preprocessing(img):
        img = grayscale(img)
        img = equalize(img)
        img = img/255
        return img

    def getCalssName(classNo):
        if   classNo == 0: return 'Speed Limit 20 km/h'
        elif classNo == 1: return 'Speed Limit 30 km/h'
        elif classNo == 2: return 'Speed Limit 50 km/h'
        elif classNo == 3: return 'Speed Limit 60 km/h'
        elif classNo == 4: return 'Speed Limit 70 km/h'
        elif classNo == 5: return 'Speed Limit 80 km/h'
        elif classNo == 6: return 'End of Speed Limit 80 km/h'
        elif classNo == 7: return 'Speed Limit 100 km/h'
        elif classNo == 8: return 'Speed Limit 120 km/h'
        elif classNo == 9: return 'No passing'
        elif classNo == 10: return 'No passing for vechiles over 3.5 metric tons'
        elif classNo == 11: return 'Right-of-way at the next intersection'
        elif classNo == 12: return 'Priority road'
        elif classNo == 13: return 'Yield'
        elif classNo == 14: return 'Stop'
        elif classNo == 15: return 'No vechiles'
        elif classNo == 16: return 'Vechiles over 3.5 metric tons prohibited'
        elif classNo == 17: return 'No entry'
        elif classNo == 18: return 'General caution'
        elif classNo == 19: return 'Dangerous curve to the left'
        elif classNo == 20: return 'Dangerous curve to the right'
        elif classNo == 21: return 'Double curve'
        elif classNo == 22: return 'Bumpy road'
        elif classNo == 23: return 'Slippery road'
        elif classNo == 24: return 'Road narrows on the right'
        elif classNo == 25: return 'Road work'
        elif classNo == 26: return 'Traffic signals'
        elif classNo == 27: return 'Pedestrians'
        elif classNo == 28: return 'Children crossing'
        elif classNo == 29: return 'Bicycles crossing'
        elif classNo == 30: return 'Beware of ice/snow'
        elif classNo == 31: return 'Wild animals crossing'
        elif classNo == 32: return 'End of all speed and passing limits'
        elif classNo == 33: return 'Turn right ahead'
        elif classNo == 34: return 'Turn left ahead'
        elif classNo == 35: return 'Ahead only'
        elif classNo == 36: return 'Go straight or right'
        elif classNo == 37: return 'Go straight or left'
        elif classNo == 38: return 'Keep right'
        elif classNo == 39: return 'Keep left'
        elif classNo == 40: return 'Roundabout mandatory'
        elif classNo == 41: return 'End of no passing'
        elif classNo == 42: return 'End of no passing by vechiles over 3.5 metric tons'
        

     
    
   
        
            
    while True:    
    # READ IMAGE
       
     # تحديد المنطقة المهتمة

    # PROCESS IMAGE
        img = np.asarray(imgOrignal)
        img = cv2.resize(img,(32,32))
        img = preprocessing(img)
        #cv2.imshow("Processed Image", img)
        img = img.reshape(1, 32, 32, 1)
       
    # PREDICT IMAGE
        predictions = model.predict(img)
       
        classIndex =np.argmax(predictions)
       # print (type(classIndex))
        probabilityValue =np.amax(predictions)
        if probabilityValue > threshold:
            print(getCalssName(classIndex))
            print(classIndex)
            print(round(probabilityValue*100,2))
            cv2.putText(imgOrignal,str(classIndex)+" "+ str(getCalssName(classIndex)), (170, 35), font, 0.50, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (200, 75), font, 0.50, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Result", imgOrignal)
        break
        

           
    
        
        
        
        
            
       
        
        

# استدعاء البرنامجين
video = "C:/Users/Hako/Desktop/traffic signs/InShot_20230416_125730078.mp4"
locations = detect_traffic_signs(video)
classify_traffic_signs(locations)
