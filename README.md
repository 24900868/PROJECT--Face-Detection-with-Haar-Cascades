# Face Detection using Haar Cascades with OpenCV and Matplotlib
# Name : Mahalakshmi M
# Reg no : 212224230148

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows

## Program
```
# Name : Iniya E
# Reg no : 212224230096

import cv2
import matplotlib.pyplot as plt
%matplotlib inline

withglass = cv2.imread('/content/Screenshot 2025-11-13 212818.png', 0)
group = cv2.imread('/content/Screenshot 2025-11-13 213045.png', 0)

plt.imshow(withglass, cmap='gray')
plt.title("With Glasses")
plt.show()

plt.imshow(group, cmap='gray')
plt.title("Group Image")
plt.show()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

if face_cascade.empty():
    raise IOError("Error loading face cascade XML file")
if eye_cascade.empty():
    raise IOError("Error loading eye cascade XML file")

def detect_face(img, scaleFactor=1.1, minNeighbors=5):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    return face_img

def detect_eyes(img):
    face_img = img.copy()
    eyes = eye_cascade.detectMultiScale(face_img)
    for (x, y, w, h) in eyes:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    return face_img

result_withglass_faces = detect_face(withglass)
plt.imshow(result_withglass_faces, cmap='gray')
plt.title("Faces in With Glasses Image")
plt.show()

result_group_faces = detect_face(group)
plt.imshow(result_group_faces, cmap='gray')
plt.title("Faces in Group Image")
plt.show()

result_withglass_eyes = detect_eyes(withglass)
plt.imshow(result_withglass_eyes, cmap='gray')
plt.title("Eyes in With Glasses Image")
plt.show()

result_group_eyes = detect_eyes(group)
plt.imshow(result_group_eyes, cmap='gray')
plt.title("Eyes in Group Image")
plt.show()
```
## Output
<img width="489" height="516" alt="image" src="https://github.com/user-attachments/assets/e517a588-0fa0-4c10-b642-241cd20d8ae2" />

<img width="582" height="513" alt="image" src="https://github.com/user-attachments/assets/b5f086df-bb14-48fd-aba0-b581ba19ccd9" />

<img width="562" height="524" alt="Screenshot 2026-06-05 094553" src="https://github.com/user-attachments/assets/c782e3d0-b006-4b9e-9215-496886d1062c" />

<img width="585" height="529" alt="image" src="https://github.com/user-attachments/assets/663121b3-7a8d-4c2d-8dda-d7e2ddc51388" />

<img width="511" height="513" alt="image" src="https://github.com/user-attachments/assets/d9b989e7-c311-4a3c-85f6-10a0165bf9a8" />

<img width="476" height="521" alt="image" src="https://github.com/user-attachments/assets/a5b90619-9f0f-4692-8d02-f6bb3daec427" />

<img width="432" height="516" alt="image" src="https://github.com/user-attachments/assets/e7c7466e-08ef-49c9-b1cd-d22749438776" />

<img width="486" height="531" alt="image" src="https://github.com/user-attachments/assets/84f8a50c-c147-426e-920a-992d21ffae3d" />

<img width="553" height="532" alt="image" src="https://github.com/user-attachments/assets/f706d9e2-7d97-4216-9199-a622b9b60652" />

<img width="715" height="533" alt="image" src="https://github.com/user-attachments/assets/7b52415e-660a-4c3a-b6a2-7e5b4d638b12" />


## Result

Thus executed successfully.
