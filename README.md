## <a href="https://github.com/secaids/dip/#ReadWrite-Image">Read & Write Image</a>
## <a href="https://github.com/secaids/dip/#Image-Acquisition-from-Web-Camera">Image Acquisition from Web Camera</a>
## <a href="https://github.com/secaids/dip/#Color-Conv">Color Conversion</a>
## <a href="https://github.com/secaids/dip/#Histogram">Histogram and Histogram Equalization of an image</a>
## <a href="https://github.com/secaids/dip/#Image-Transformation">Image Transformation</a>
## <a href="https://github.com/secaids/dip/#Filters">Implemetation of Filters</a>
## <a href="https://github.com/secaids/dip/#"></a>
## <a href="https://github.com/secaids/dip/#"></a>
## <a href="https://github.com/secaids/dip/#"></a>
## <a href="https://github.com/secaids/dip/#"></a>
## <a href="https://github.com/secaids/dip/#"></a>
## <a href="https://github.com/secaids/dip/#"></a>

## Read&Write-Image
i) Read and display the image
```py
import cv2
img = cv2.imread("dipt.png")
cv2.imshow("read_pic",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
ii) To write the image
```
cv2.imwrite("write_pic.png",img)
```
iii) Find the shape of the Image
```py
print(img.shape)
```
iv) To access rows and columns
```py
for i in range(350,400):
    for j in range(800,1000):
        img[i][j] = [104, 104, 104]
cv2.imshow("row_pic.png",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
v) To cut and paste portion of image
```py
img[700:1000,600:900] = img[300:600,1100:1400]
cv2.imshow("cut_pic.png",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
## Image-Acquisition-from-Web-Camera
i) Write the frame as JPG file
```py
obj = cv2.VideoCapture(0)
while(True):
    cap,frame = obj.read()
    cv2.imshow('video_image.jpg',frame)
    cv2.imwrite("out.jpg",frame)
    if cv2.waitKey(1) == ord('q'):
        break
obj.release()
cv2.destroyAllWindows()
```
ii) Display the video
```py
obj = cv2.VideoCapture(0)
while(True):
    cap,frame = obj.read()
    cv2.imshow('video_image',frame)
    if cv2.waitKey(1) == ord('q'):
        break
obj.release()
cv2.destroyAllWindows()
```
iii) Display the video by resizing the window
```py
cap=cv2.VideoCapture(0)
while(True):
    ret,frame=cap.read()
    width=int(cap.get(3))
    height=int(cap.get(4))
    
    image=np.zeros(frame.shape,np.uint8)
    smaller_frame = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
    image[:height//2,:width//2]=smaller_frame
    image[height//2:,:width//2]=smaller_frame
    image[:height//2,width//2:]=smaller_frame
    image[height//2:,width//2:]=smaller_frame
    
    cv2.imshow('quadrant_screen',image)
    if cv2.waitKey(1) == ord('q'):
        break
VidCap.release()
cv2.destroyAllWindows()
```
iv) Rotate and display the video
```py
cap2=cv2.VideoCapture(0)
while(True):
    ret,frame=cap.read()
    width=int(cap.get(3))
    height=int(cap.get(4))
    
    image=np.zeros(frame.shape,np.uint8)
    smaller_frame = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
    image[:height//2,:width//2]=cv2.rotate(smaller_frame,cv2.ROTATE_180)
    image[height//2:,:width//2]=smaller_frame
    image[:height//2,width//2:]=cv2.rotate(smaller_frame,cv2.ROTATE_180)
    image[height//2:,width//2:]=smaller_frame
    
    cv2.imshow('quadrant_rotated_screen',image)
    if cv2.waitKey(1) == ord('q'):
        break
cap2.release()
cv2.destroyAllWindows()
```
## Color-Conv
**i) Convert BGR and RGB to HSV and GRAY**
```python
image=cv2.imread("original.PNG",1)
img= cv2.resize(image, (465,324))
cv2.imshow("original",img)

hsv_bgr=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
cv2.imshow("BGR to HSV",hsv_bgr)

hsv_rgb= cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
cv2.imshow('RGB to HSV',hsv_rgb)

gray_bgr= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('BGR2GRAY',gray_bgr)

gray_rgb= cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
cv2.imshow('RGB2GRAY',gray_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()
```
**ii)Convert HSV to RGB and BGR**
```python
image2 = cv2.imread("original.PNG")
img2= cv2.resize(image2, (465,324))

hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
cv2.imshow("hsv_image", hsv)

hsv_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
cv2.imshow("hsv to rgb", hsv_rgb)

hsv_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imshow("hsv to bgr", hsv_bgr)

cv2.waitKey(0)
cv2.destroyAllWindows()
```
**iii)Convert RGB and BGR to YCrCb**
```python
image3 = cv2.imread("original4.PNG")
img3= cv2.resize(image3, (470,324))

cv2.imshow("original(bgr)", img3)
img_ycrcb = cv2.cvtColor(img3 , cv2.COLOR_BGR2YCrCb)
cv2.imshow("bgr to YCrCb ", img_ycrcb)

img_rgb = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
cv2.imshow("rgb", img_rgb)
img_bgr_y = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2YCrCb)
cv2.imshow("rgb to YCrCb", img_bgr_y)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
**iv)Split and Merge RGB Image**
```python
image4 = cv2.imread("original3.PNG")
img4= cv2.resize(image4, (470,324))

b,g,r = cv2.split(img4)
cv2.imshow("red model", r)
cv2.imshow("green model", g)
cv2.imshow("blue model ", b)

merger = cv2.merge([b,g,r])
cv2.imshow("merged", merger )
cv2.waitKey(0)
cv2.destroyAllWindows()
```
**v) Split and merge HSV Image**
```python
image5 = cv2.imread("original5.PNG")
img5= cv2.resize(image5, (470,294))

hsv = cv2.cvtColor(img5 , cv2.COLOR_BGR2HSV)
cv2.imshow("initial hsv ", hsv)

h,s,v = cv2.split(hsv)
cv2.imshow("hue model", h)
cv2.imshow("saturation model", s)
cv2.imshow("value model ", v)

merger = cv2.merge([h,s,v])
cv2.imshow("merged image", merger )
cv2.waitKey(0)
cv2.destroyAllWindows()
```
## Histogram

### Histogram of gray scale image and color image channels.
```python
import cv2
import matplotlib.pyplot as plt

gray=cv2.imread("grey_referece.PNG",0)
gray= cv2.resize(gray, (627,403))
cv2.imshow('gray image',gray)

color=cv2.imread("color_reference.PNG",1)
color= cv2.resize(color, (627,415))
cv2.imshow('color image',color)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Histogram of gray scale image and any one channel histogram from color image
```python
gray_hist=cv2.calcHist([gray],[0],None,[256],[0,255])
color_hist=cv2.calcHist([color],[2],None,[256],[0,255])

plt.figure()
plt.title("gray image")
plt.xlabel("grayscale value")
plt.ylabel("pixel count")
plt.stem(gray_hist)
plt.show()

plt.figure()
plt.title("color image")
plt.xlabel("colorscale value")
plt.ylabel("pixel count")
plt.stem(color_hist)
plt.show()
```

### Histogram equalization of the image. 
```python
gray_equalized=cv2.equalizeHist(gray)

cv2.imshow('gray image',gray)
cv2.imshow('equalized gray image',gray_equalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
## Image-Transformation
**Image Translation**
```python
import numpy as np
import cv2

in_img=cv2.imread("dip.jpg",1)

#getting rows,columns,hues(if any)
rows,cols,dim=in_img.shape

#image translation
m=np.float32([[1,0,150],
              [0,1,100],
              [0,0,1]])
trans_img=cv2.warpPerspective(in_img,m,(cols,rows))

cv2.imshow("original image",in_img)
cv2.imshow("translated image",trans_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

**Image Scaling**
```python
m=np.float32([[1.5,0,0],
              [0,1.8,0],
              [0,0,1]])
scale_img=cv2.warpPerspective(in_img,m,(cols,rows))

cv2.imshow("original image",in_img)
cv2.imshow("scaled image",scale_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

**Image shearing**
```python
m_x=np.float32([[1,0.2,0],
                [0,1,0],[
                0,0,1]])
m_y=np.float32([[1,0,0],
                [0.4,1,0],
                [0,0,1]])

sheared_img_x=cv2.warpPerspective(in_img,m_x,(cols,rows))
sheared_img_y=cv2.warpPerspective(in_img,m_y,(cols,rows))

cv2.imshow("original image",in_img)
cv2.imshow("sheared img x-axis",sheared_img_x)
cv2.imshow("sheared img y-axis",sheared_img_y)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

**Image Reflection**
```python
in2_img=cv2.imread("maki.jpg",1)
rows2,cols2,dim2=in2_img.shape

m_x=np.float32([[1,0,0],
                [0,-1,rows2],
                [0,0,1]])
m_y=np.float32([[-1,0,cols2],
                [0,1,0],
                [0,0,1]])
reflected_img_x=cv2.warpPerspective(in2_img,m_x,(cols2,rows2))
reflected_img_y=cv2.warpPerspective(in2_img,m_y,(cols2,rows2))

cv2.imshow("original image",in2_img)
cv2.imshow("reflected img x-axis",reflected_img_x)
cv2.imshow("reflected img y-axis",reflected_img_y)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

**Image Rotation**
```python
angle=np.radians(45)
m=np.float32([[np.cos(angle),-(np.sin(angle)),0],
              [np.sin(angle),np.cos(angle),0],
              [0,0,1]])
rotated_img=cv2.warpPerspective(in2_img,m,(cols,rows))

cv2.imshow("original image",in2_img)
cv2.imshow("rotated image",rotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

**Image Cropping**
```python
crop_img=in2_img[100:400,100:300]

cv2.imshow("original image",in2_img)
cv2.imshow("cropped image",crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
## Filters
### 1. Smoothing Filters
**Using Averaging Filter**
```Python
import cv2
import matplotlib.pyplot as plt
import numpy as np

orig=cv2.imread("dipt.jpg")

kernel=np.ones((11,11),np.float32)/121
avg=cv2.filter2D(orig,-1,kernel)
cv2.imshow('original image',orig)
cv2.imshow('averaging filtered image',avg)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
**Using Weighted Averaging Filter**
```Python
kernel2=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
w_avg=cv2.filter2D(orig,-1,kernel2)
cv2.imshow('original image',orig)
cv2.imshow('weighted averaging filtered image',w_avg)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
**Using Gaussian Filter**
```Python
gauss=cv2.GaussianBlur(orig,(33,33),0,0)
cv2.imshow('original image',orig)
cv2.imshow('gaussian blurred image',gauss)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

**Using Median Filter**
```Python
median=cv2.medianBlur(orig,13)
cv2.imshow('original image',orig)
cv2.imshow('median blurred image',median)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 2. Sharpening Filters
**Using Laplacian Kernal**
```Python
kernel3=np.array([[-1,-1,-1],[2,-2,1],[2,1,-1]])
lap_k=cv2.filter2D(orig,-1,kernel3)
cv2.imshow('original image',orig)
cv2.imshow('Laplacian Kernel filtered image',lap_k)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
**Using Laplacian Operator**
```Python
lap_o=cv2.Laplacian(orig,cv2.CV_64F)
cv2.imshow('original image',orig)
cv2.imshow('Laplacian Operator filtered image',lap_o)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
