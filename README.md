# autonomous_rc_car
<img width="50%" src="https://user-images.githubusercontent.com/74947395/165596921-fab06ed2-a041-4788-a37b-0a86ba74ece9.jpg"/>

## Raspi: detect car lane
<img width="50%" src="https://user-images.githubusercontent.com/74947395/165596098-2d16e0e8-af50-4f0f-993b-a5c882b22355.png"/>
1) yellow & white : 범위 설정 - detect
    
    lower_white=(200,200,200)
    upper_white=(255,255,255)

    lower_yellow = (10, 100, 100)
    upper_yellow = (40, 255, 255)
    
<img width="50%" src="https://user-images.githubusercontent.com/74947395/165595947-bfa97c22-c90e-4fe9-bf88-be2774853b52.png"/>
2) gaussian & canny: find contour
<img width="50%" src="https://user-images.githubusercontent.com/74947395/165596015-1a1724ad-7ddb-4b59-b315-538c5d59c657.png"/>
3) choose ROI: 도로만 감지
4) Hough line detect
 
 lines=cv2.HoughLinesP(roi,1,np.pi/180,10,None,120,100)

5) Seperate left & right Line: detect 
6) regression by vanishing point
7) predict direction


## Raspi: detect traffic sign 
<img width="50%" src="https://user-images.githubusercontent.com/74947395/165594075-397350a1-0162-4854-b39c-1857efc5cd1e.png"/>
1) Yolo v3: [up, down, ternel, rotary, curve, light, parking]
-> 인식률 좋으나 실시간 구동 힘듦
2) use ORB template matching


## Arduino : RC car with bluetooth module
