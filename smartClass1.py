import cv2
import pandas as pd
from ultralytics import YOLO

model=YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)

def determine_zone(x, y, line1, line2):
    x1, y1 = line1['xs'], line1['ys']
    x2, y2 = line2['xs'], line2['ys']
    x3, y3 = line1['xe'], line1['ye']
    x4, y4 = line2['xe'], line2['ye']
    m1, b1 = slope_intercept(line1['xs'], line1['ys'], line1['xe'], line1['ye'])
    m2, b2 = slope_intercept(line2['xs'], line2['ys'], line2['xe'], line2['ye'])

    side_line1 = (y - m1 * x - b1) * (x3 - x1) - (x - x1) * (y3 - y1)
    side_line2 = (y - m2 * x - b2) * (x4 - x2) - (x - x2) * (y4 - y2)

    if side_line1 > 0:
        if side_line2 > 0:
            return 1
        else:
            return 2
    else:
        if side_line2 > 0:
            return 3
        else:
            return 4
def slope_intercept(x1, y1, x2, y2):
    slope = (y2-y1)/(x2-x1)
    intercept = y1 - (slope*x1)
    return slope, intercept

# def intersecting_point(line1, line2):
#     m1, b1 = slope_intercept(line1['xs'], line1['ys'], line1['xe'], line1['ye'])
#     m2, b2 = slope_intercept(line2['xs'], line2['ys'], line2['xe'], line2['ye'])
#     x = (b2-b1)/(m1-m2)
#     y = m1*x +b1
#     return x,y

# def expected_coord(line):
#     m, b = slope_intercept(line['xs'], line['ys'], line['xe'], line['ye'])
    

cv2.namedWindow('Class')
cv2.setMouseCallback('Class', RGB)

cap=cv2.VideoCapture('vid1.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

# cy1=281
# cy2=374
# offset = 3 

ln_coord_1 = {'xs':214, 'ys':303, 'xe':780, 'ye':499}

# a1, b1 = slope_intercept(ln_coord_1['xs'], ln_coord_1['ys'], ln_coord_1['xe'], ln_coord_1['ye'])
# expected_y_1 = lambda x: a1*x + b1

ln_coord_2 = {'xs':623, 'ys':289, 'xe':251, 'ye':497}

# a2, b2 = slope_intercept(ln_coord_1['xs'], ln_coord_1['ys'], ln_coord_1['xe'], ln_coord_1['ye'])
# expected_x_2 = lambda y: (y-b2)/a2
# inter_x, inter_y = intersecting_point(ln_coord_1, ln_coord_2)

count = 0
while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 10 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
   
    results=model.predict(frame)

    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")

    detected_person=[]
    strin = ""
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'person' in c:
            detected_person.append([x1,y1,x2,y2])
    for idx in detected_person: 
        cx=int(idx[0]+idx[2])//2
        cy=idx[3]
        
        

        cv2.rectangle(frame, (idx[0], idx[1]), (idx[2], idx[3]), (0, 255, 0), 2)
        cv2.putText(frame, '.', (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        strin = "Zone " + str(determine_zone(cx, cy, ln_coord_1, ln_coord_2))
        # print(strin)
        # if inter_y>cy:
        #     if inter_x>cx:
        #         strin="Zone 1"
        #     elif inter_x<cx:
        #         strin="Zone 2"
        # elif inter_y<cy:
        #     if inter_x>cx:
        #         strin="Zone 3"
        #     elif inter_x<cx:
        #         strin="Zone 4"


    cv2.line(frame,(ln_coord_1['xs'], ln_coord_1['ys']),(ln_coord_1['xe'], ln_coord_1['ye']),(255,255,255),1)
    cv2.putText(frame,str(strin),(151, 112),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)

    cv2.line(frame,(ln_coord_2['xs'], ln_coord_2['ys']),(ln_coord_2['xe'], ln_coord_2['ye']),(255,255,255),1)

    cv2.imshow("Class", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
