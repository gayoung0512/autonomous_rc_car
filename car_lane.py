import cv2
import numpy as np
import math
import itertools

cap = cv2.VideoCapture(-1)

trap_bottom_width=1.7#사다리꼴 아래쪽 가장자리 너비 계산 위한 백분율
trap_top_width=0.65 # 사다리꼴 위쪽 가장자리 너비 계산을 위한 백분율
trap_height=0.45# 사다리꼴 높이 계산을 위한 백분율

#관심 영역 계산 시 사용
#car_test_4
#trap_bottom_width=1.8#사다리꼴 아래쪽 가장자리 너비 계산 위한 백분율
#trap_top_width=0.6 # 사다리꼴 위쪽 가장자리 너비 계산을 위한 백분율
#trap_height=0.6# 사다리꼴 높이 계산을 위한 백분율

#좌우영역 구분 시 사용
slope_thresh_high=5.0
#slope_thresh_low =0.8 # 기울기 기준
slope_thresh_low =0.2
slopes = []
selected_lines = []
right_lines = []
left_lines = []
left_detect=0
right_detect=0

left_pts=[]
right_pts=[]

right_m=2
right_b=[2,2]
left_m=1
left_b=[1,1]

right_ini_x=0
right_fni_x=0
poly_points=[]

#색상 검출
def filter_colors(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #흰색 범위
    lower_white=(170,170,170)
    upper_white=(255,255,255)

    #노란색 범위
    lower_yellow = (0, 60, 60)
    upper_yellow = (40, 180, 180)

    img_mask_white = cv2.inRange(img, lower_white, upper_white)
    img_mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    #팽창 연산 수행 (원본 배열, 연산 방법, 구조 요소, 고정점, 반복 횟수)
    img_mask_yellow= cv2.morphologyEx(img_mask_yellow, cv2.MORPH_DILATE, kernel, iterations=3)
    img_mask_white= cv2.morphologyEx(img_mask_white, cv2.MORPH_DILATE, kernel, iterations=3)

    img_mask = cv2.addWeighted(img_mask_white, 0.0, img_mask_yellow, 1.0, 0.0)
    # 구조 요소 생성 (커널의 형태[직사각형],커널의 크기)

    #cv2.imshow('mask_yellow', img_mask_yellow)
    #cv2.imshow('mask_white', img_mask_white)
    #cv2.imshow('mask', img_mask)
    #cv2.imshow('Result', img)
    return img_mask

def limit_region(canny):#자동차의 진행 방향 바닥에 존재하는 직선만을 검출 -> 관심영역 지정
    h,w=canny.shape
    h=int(h)
    w=int(w)
    mask=np.zeros((h,w),np.uint8)
    points=np.array([[int(w*(1-trap_bottom_width)/2),h],[int(w*(1-trap_top_width)/2),int(h-h*trap_height)],[int(w-(w*(1-trap_top_width)/2)),int(h-h*trap_height)],[int(w-(w*(1-trap_bottom_width))/2),h]])
    blue=(255,0,0)
    mask=cv2.fillConvexPoly(mask,points,blue)#mask에 영역 적기
    roi=cv2.bitwise_and(canny,mask)
    #cv2.imshow('mask',mask)

    roi_mask = cv2.addWeighted(mask, 0.5, canny, 0.5, 0.0)#관심영역 -> 가중치 설정해서 확인
    #cv2.imshow('roi_mask',roi_mask)
    return roi

def Hough(roi,img):#허프 변환
    lines=cv2.HoughLinesP(roi,1,np.pi/180,10,None,80,20)#직선 좌표 저장
    #최대 길이, 최소 간격
    for line in lines:
        x1,y1,x2,y2=line[0]
        cv2.line(roi,(x1,y1),(x2,y2),(120,120,120),10)
        cv2.line(img, (x1, y1), (x2, y2), (120, 120, 120), 10)
    return img,roi,lines
def seperate_line(hough,image,lines):#추출한 직선 성분으로 좌우 차선에 있을 가능성이 있는 직선만 뽑아서 좌우 직선 검출
    slopes=[]# 기울기 저장
    selected_lines=[] #기울기가 적합하다고 판단되는 직선 저장
    right_lines=[]
    left_lines=[]
    right_detect=0
    left_detect=0
    h, w = hough.shape
    global img
    for line in lines:
        x1,y1,x2,y2=line[0]
        slope=(y2-y1)/(x2-x1)#기울기 계산
        try:
            if abs(slope) > slope_thresh_low and abs(slope) < slope_thresh_high:#기울기가 수평인 선 제외
                slopes.append(slope)
                selected_lines.append(line)
                #img = cv2.line(img, (x1,y1), (x2,y2), (0, 120, 255), 10, cv2.LINE_AA) #주황
        except ZeroDivisionError: #error 떠도 종료되지 않게 설정
            print('Zero')

    #좌우선으로 분류
    img_center=w/2
    l=len(selected_lines)
    for i in range(0,l):
        x3,y3,x4,y4=selected_lines[i][0] #기울기만 최적화
        if (slopes[i]>0):# and (x4>=img_center)and(x3>=img_center):
            right_lines.append(selected_lines[i][0])
            right_detect=1
            #x1, y1, x2, y2 = selected_lines[i][0]
            #img = cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), 10, cv2.LINE_AA)
        if slopes[i] < 0 :#and x4 <= img_center and x3 <= img_center:
            left_lines.append(selected_lines[i][0])
            #x1, y1, x2, y2 = selected_lines[i][0]
            #img = cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 10, cv2.LINE_AA)
            left_detect=1
    output=(right_lines,left_lines)
    #cv2.imshow('right', img)
    return output,image,right_detect,left_detect

#최적화된 선 찾아내기
def regression(sl,image,right_detect,left_detect):
    right_pts=[]
    left_pts=[]
    global left_m,left_b,right_m,right_b

    if right_detect == 1:
        print('right')
        for s in sl[0]: #sl[0]=output[0]=right_lines
            x5,y5,x6,y6=s #오른쪽 직선 좌표
            right_pts.append([x5,y5])
            right_pts.append([x6,y6])
        if len(right_pts)>0:
            right_pts=np.asarray(right_pts)
            right_line = cv2.fitLine(right_pts, cv2.DIST_L2, 0, 0.01, 0.01)#근사화
            right_m=float(float(right_line[1])/float(right_line[0]))#기울기
            right_b=(right_line[2],right_line[3])#점

    if left_detect == 1:
        print('left')
        for s in sl[1]:
            x7,y7,x8,y8=s
            left_pts.append([x7,y7])
            left_pts.append([x8,y8])
        #왼쪽 선들 근사화해 최적화된 왼쪽 선 찾음
        if len(left_pts)>0:
            left_pts=np.asarray(left_pts)
            left_line = cv2.fitLine(left_pts, cv2.DIST_L2, 0, 0.01, 0.01)#근사화
            left_m=float(float(left_line[1])/float(left_line[0]))#기울기
            left_b=(left_line[2],left_line[3])#점

    h, w ,_ = image.shape
    h = int(h)
    w = int(w)

    #좌우 선 각각의 두 점 계산
    #y=m*x+b-->x=(y-b)/m
    #400은 계산을 위해 임의로 설정

    right_ini_x=int(((h-right_b[1])/right_m)+right_b[0])
    right_fni_x = int(((400 - right_b[1]) / right_m) + right_b[0])
    left_ini_x=int(((h-left_b[1])/left_m)+left_b[0])
    left_fni_x = int(((400 - left_b[1]) / left_m) + left_b[0])

    output=[[right_ini_x,h], [right_fni_x,400],[left_ini_x,h],[left_fni_x,400]]
    #image = cv2.line(image, (right_ini_x,h), (right_fni_x,300), (0, 50, 50), 30, cv2.LINE_AA)
    #image = cv2.line(image, (left_ini_x,h),(left_fni_x,300), (0, 255, 255), 20, cv2.LINE_AA)
    return output,image

#두 차선 교차하는 지점이 중심점으로부터 좌우 어느쪽에 있는지
def predictDir(image,left_detect,right_detect):#진행방향 예측
    h, w ,_ = image.shape
    h = int(h)
    w = int(w)
    img_center=w/2
    #car_test_3
    #thres_vp=80
    thres_vp=80
    vx=float(((right_m*right_b[0])-(left_m*left_b[0])-right_b[1]+left_b[1])/(right_m-left_m))
    if left_detect==0 and right_detect==1:
        output='Left Turn'
    elif left_detect==1 and right_detect==0:
        output='Right turn'
    else:
        if vx<(img_center-thres_vp):
            output='Left Turn'
        elif vx>(img_center+thres_vp) :
            output='Right Turn'
        elif (vx>=(img_center-thres_vp))&(vx<=(img_center+thres_vp)):
            output='Straight'

    return output
def drawLine(img, lane, dir):
    poly_points=[]
    poly_points.append(lane[2])
    poly_points.append(lane[0])
    poly_points.append(lane[1])
    poly_points.append(lane[3])
    img_copy=img
   # img=cv2.fillConvexPoly(img, poly_points,(0,230,30),cv2.LINE_AA,0)
    #print(lane)
    #print('lane\n\n\n\n')
    p0=[int(lane[0][0]),int(lane[0][1])]
    p1 = [int(lane[1][0]), int(lane[1][1])]
    p2=[int(lane[2][0]),int(lane[2][1])]
    p3 = [int(lane[3][0]), int(lane[3][1])]

    img=cv2.addWeighted(img_copy,0.3,img,0.7,0.0)
    #img=cv2.line(img, p0,p1,(0,255,255),10,cv2.LINE_AA)
    #img=cv2.line(img, p2,p3,(0,255,255),10,cv2.LINE_AA)

    img=cv2.putText(img, dir, (520,100),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3,cv2.LINE_AA)
    return img
while True:
    ret, img = cap.read()#실시간 영상 읽어오기
    #img=cv2.resize(img, dsize=(500, 500))
    img_mask=filter_colors(img) #white & yellow filtering
    blur = cv2.GaussianBlur(img_mask, (3, 3), 0)#가우시안 필터 - 부드럽게 보이게 만들어줌 : 노이즈 제거
    canny = cv2.Canny(blur, 100, 200) # canny 함수 - 윤곽선 검출
    roi=limit_region(canny) #ROI 관심영역
    img,hough,lines=Hough(roi,img) #직선 검출

    # 좌우 차선 구분 & 방향 인식
    if lines.size>0:
         sl,slimg,rd,ld=seperate_line(hough,img,lines)#좌우 분리
         lane,rimg=regression(sl,img,rd,ld)#진행방향 예측
         dir=predictDir(img,ld,rd)
         result=drawLine(img,lane, dir)
    #cv2.imshow('mask', img_mask)
    #cv2.imshow('gaussian', blur)
    #cv2.imshow('image',img)
    #cv2.imshow('canny',canny)
    key = cv2.waitKey(1)
    #cv2.imshow('line',roi)
    cv2.imshow('hough',hough)
    #cv2.imshow('rimg',rimg)
    cv2.imshow('result',result)
    if key == 27:  # esc
        break
