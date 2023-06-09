## <a href="https://github.com/secaids/dip/#ReadWrite-Image">Read & Write Image</a>
## <a href="https://github.com/secaids/dip/#Web-Camera">Image Acquisition from Web Camera</a>
## <a href="https://github.com/secaids/dip/#Color-Conv">Color Conversion</a>
## <a href="https://github.com/secaids/dip/#Histogram">Histogram and Histogram Equalization of an image</a>
## <a href="https://github.com/secaids/dip/#Transformation">Image Transformation</a>
## <a href="https://github.com/secaids/dip/#Filters">Implemetation of Filters</a>
## <a href="https://github.com/secaids/dip/#Edge">Edge Detection</a>
## <a href="https://github.com/secaids/dip/#hough-transform">Edge Linking using Hough Transform</a>
## <a href="https://github.com/secaids/dip/#Thresholding">Thresholding of Images</a>
## <a href="https://github.com/secaids/dip/#erosion-dilation">Erosion and Dilation</a>
## <a href="https://github.com/secaids/dip/#opening-closing">Opening and Closing</a>
## <a href="https://github.com/secaids/dip/#huffman">Huffman Coding</a>

## Read&Write Image
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
## Web Camera
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
## Color Conv
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
## Transformation
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
## Edge
**Load the image, Convert to grayscale and remove noise**
```py
input_img = cv2.imread("mikasa.jpg",1)

gray = cv2.cvtColor(input_img,COLOR_BGR2GRAY)

img = cv2.GaussianBlur(gray,(3,3),0)

cv2.imshow("GaussianBlur",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

**SOBEL EDGE DETECTOR**
```py
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
sobelxy =cv2.Sobel(img,cv2.CV_64F,1,1,ksize=5)

plt.imshow(img,cmap = 'gray')
plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(sobelx,cmap = 'gray')
plt.title('sobelx')
plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(sobely,cmap = 'gray')
plt.title('sobely')
plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(sobelxy,cmap = 'gray')
plt.title('sobelxy')
plt.xticks([]), plt.yticks([])

plt.show()
```
**LAPLACIAN EDGE DETECTOR**
```py
laplacian = cv2.Laplacian(img,cv2.CV_64F)
cv2.imshow("laplacian",laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

**CANNY EDGE DETECTOR**
```py
canny_edges = cv2.Canny(img, 120, 150)
cv2.imshow("Canny",canny_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
## Hough Transform
```py
### Import Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

### Read Image
image=cv2.imread("dipt.png",1)

plt.imshow(image)
plt.title('Original')
plt.axis('off')

### Convert image to grayscale
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

plt.imshow(image,cmap="gray")
plt.title('gray')
plt.axis('off')


### Smoothen image using Gaussian Filter
img = cv2.GaussianBlur(src = gray, ksize = (15,15), sigmaX=0,sigmaY=0)

plt.imshow(img)
plt.title('EDGES')
plt.axis('off')

### Find the edges in the image using canny detector
edges = cv2.Canny(image, 120, 150)
plt.imshow(edges)
plt.title('EDGES')
plt.axis('off')

### Detect points that form a line using HoughLinesP
lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=80,minLineLength=50,maxLineGap=250)

### Draw lines on the image
for line in lines:
    x1,y1,x2,y2=line[0]
    cv2.line(image,(x1,y1),(x2,y2),(80,0,50),2)

### Display the result
plt.imshow(image)
plt.title('HOUGH')
plt.axis('off')
```
## Thresholding
### Load the necessary packages
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```
### Read the Image and convert to grayscale
```python
in_img=cv2.imread('dip.PNG')
in_img2=cv2.imread('diptPNG')

gray_img = cv2.cvtColor(in_img,cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(in_img2,cv2.COLOR_BGR2GRAY)
```
### Use Global thresholding to segment the image
```python
# cv2.threshold(image, threshold_value, max_val, thresholding_technique)
ret,thresh_img1=cv2.threshold(gray_img,86,255,cv2.THRESH_BINARY)
ret,thresh_img2=cv2.threshold(gray_img,86,255,cv2.THRESH_BINARY_INV)
ret,thresh_img3=cv2.threshold(gray_img,86,255,cv2.THRESH_TOZERO)
ret,thresh_img4=cv2.threshold(gray_img,86,255,cv2.THRESH_TOZERO_INV)
ret,thresh_img5=cv2.threshold(gray_img,100,255,cv2.THRESH_TRUNC)
```
### Use Adaptive thresholding to segment the image
```python
# cv2.adaptiveThreshold(source, max_val, adaptive_method, threshold_type, blocksize, constant)
thresh_img6=cv2.adaptiveThreshold(gray_img2,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
thresh_img7=cv2.adaptiveThreshold(gray_img2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
```
### Use Otsu's method to segment the image 
```python
# cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret,thresh_img8=cv2.threshold(gray_img2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
```
### Display the results
```python
cv2.imshow('original image',in_img)
cv2.imshow('original image(second)',in_img2)

cv2.imshow('original image(gray)',gray_img)
cv2.imshow('original image(gray)(second)',gray_img2)

cv2.imshow('binary threshold',thresh_img1)
cv2.imshow('binary-inverse threshold',thresh_img2)
cv2.imshow('to-zero threshold',thresh_img3)
cv2.imshow('to-zero-inverse threshold',thresh_img4)
cv2.imshow('truncate threshold',thresh_img5)

cv2.imshow('mean adaptive threshold',thresh_img6)
cv2.imshow('gaussian adaptive threshold',thresh_img7)

cv2.imshow('otsu\'s threshold',thresh_img8)

cv2.waitKey(0)
cv2.destroyAllWindows()
```
## Erosion Dilation
### Import the necessary packages
```py
import cv2
import numpy as np
from matplotlib import pyplot as plt
```
### Create the text using cv2.putText
```py
img1 = np.zeros((100,550), dtype = 'uint8')
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img1,'DIPT',(5,70), font, 2,(255),5,cv2.LINE_AA)
plt.imshow(img1,'gray')
```
### Create the structuring element
```py
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))
cv2.erode(img1, kernel)
```
### Erode the image
```py
image_erode1 = cv2.erode(img1,kernel)
plt.imshow(image_erode1, 'gray')
```
### Dilate the image
```py
image_dilate1 = cv2.dilate(img1, kernel)
plt.imshow(image_dilate1, 'gray')
```
## Opening Closing
### Import the necessary packages
```python 
import cv2
import numpy as np
import matplotlib.pyplot as plt
```
### Create the Text using cv2.putText
```python
text_image = np.zeros((100,250),dtype = 'uint8')
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
cv2.putText(text_image,"DIPT",(5,70),font,2,(255),2,cv2.LINE_AA) 
plt.title("Original Image")
plt.imshow(text_image,'Blues')
plt.axis('off')
```
### Create the structuring element
```python
kernel1=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
kernel2=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
```
### Use Opening operation
```python
img_open=cv2.morphologyEx(text_image,cv2.MORPH_OPEN,kernel2)
plt.title("Opened Image")
plt.imshow(img_open,'Blues')
plt.axis('off')
```
### Use Closing Operation
```python
img_close=cv2.morphologyEx(text_image,cv2.MORPH_CLOSE,kernel1)
plt.title("Closed Image")
plt.imshow(img_close,'Blues')
plt.axis('off')
```
## Huffman
### Get the input String
```py
string = "Digital Image Processing"
```
### Create tree nodes
```py
class NodeTree(object):
    def __init__(self, left=None, right=None): 
        self.left = left
        self.right=right
    def children(self):
        return (self.left,self.right)
```
### Main function to implement huffman coding
```py
def huffman_code_tree (node, left=True, binString=''):
    if type(node) is str:
        return {node: binString}
    (l, r) = node.children()
    d = dict()
    d.update(huffman_code_tree (l, True, binString + '0'))
    d.update(huffman_code_tree (r, False, binString + '1'))
    return d
```
### Calculate frequency of occurrence
```py
freq = {}
for c in string:
    if c in freq:
        freq[c] += 1
    else:
        freq[c] = 1

freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
nodes=freq

while len(nodes)>1:
    (key1,c1)=nodes[-1]
    (key2,c2)=nodes[-2]
    nodes = nodes[:-2]
    node = NodeTree (key1, key2)
    nodes.append((node,c1 + c2))
    nodes = sorted (nodes, key=lambda x: x[1], reverse=True)
```
### Print the characters and its huffmancode
```py
huffmanCode=huffman_code_tree(nodes[0][0])
print(' Char | Huffman code ') 
print('----------------------')
for (char, frequency) in freq:
    print('%-4r  |%12s'%(char,huffmanCode[char]))
```
