import numpy as np
import cv2
import face_recognition
from PIL import Image, ImageDraw

#capture the video from default camera 
webcam_video_stream = cv2.VideoCapture(0)

#loop through every frame in the video
while True:
	
  #get the current frame from the video stream as an image
  ret,current_frame = webcam_video_stream.read()
  
  #get the face landmarks list for all faces
  face_landmarks_list = face_recognition.face_landmarks(current_frame)

  #convert the numpy array image into pil image object
  pil_img = Image.fromarray(current_frame)

  # loop through face landmarks of each face
  for face_landmarks in face_landmarks_list:
	
	#convert the pil image to draw object
    d = ImageDraw.Draw(pil_img)
	
	# loop through landmark type in that particular face
    for landmak_type in face_landmarks.keys():
	  
	  #join each face landmark points
      d.line(face_landmarks[landmak_type], fill = (15, 255, 80), width= 1)

  #convert PIL image to show in opencv window 
  open_cv_image = np.array(pil_img)

  #showing the current face with landmark drawn
  cv2.imshow("Webcam Video", open_cv_image)
    
  # q to quit webcam window  
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break


#release the stream and cam
#close all opencv windows open
webcam_video_stream.release()
cv2.destroyAllWindows()       