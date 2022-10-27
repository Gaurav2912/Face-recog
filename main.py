"""
@author: Gaurav Yogeshwar
"""

#importing the required libraries
import face_recognition
import numpy as np
import cv2
import os
from datetime import datetime

# Importing Images
path = 'imgs'
known_imgs = []                          # LIST CONTAINING ALL THE IMAGES
known_names = []                          # LIST CONTAINING ALL THE CORRESPONDING  Names
known_face_encodings = []                   #  LIST CONTAINING ALL THE CORRESPONDING  known face encoding
list_dir = os.listdir(path)                      # List of images file name from directory

print("Total images Detected:", len(list_dir))

# Loop through each item in list_dir
for item in list_dir:
    name = os.path.splitext(item)[0]                           # garb the name of the file, without its format (.jpg)              
    known_names.append(name)                                   # append it to list
    curr_img = cv2.imread(path + '/' + item)                   # rerading file as in image form
    curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)        # converting from BGR to RGB  
    known_imgs.append(curr_img)                                # append it to list

    # face encoding for corresponding known face
    face_encoding =  face_recognition.face_encodings(curr_img)[0]       
    known_face_encodings.append(face_encoding)

# Marking Attendance
def markAttendance(name):   
    ''' 
    function that update name and time in csv file
    '''   
    with open('Attendance.csv','r+') as f:                  # opening in read and write mode
        my_data_line = f.readlines()                        # reading line by line
        name_list = []                                      
        for line in my_data_line:
            entry = line.split(',')
            name_list.append(entry[0])
        
        # It will update attendance only once
        if name not in name_list:
            now = datetime.now()
            dt_string = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dt_string}')


# print face distances, range of face distance is 0 to 1 
def faceDistanceCalculator(face_encodings, face_to_compare):
    '''
    calculate distance between to face encodings and percentage of match.
    '''
   # find the face distance of current_face_encoding with the known samples, return a list of distances with known images to current frame
    face_distances = face_recognition.face_distance(face_encodings, face_to_compare)

    # Minimum distance 
    minimum_distance = min(face_distances)
    
    # percentage of match 
    percent_of_match = (1 - minimum_distance) * 100


    return minimum_distance, percent_of_match



#capture the video from default camera 
webcam_video_stream = cv2.VideoCapture(0)

#loop through every frame in the video
while True:
    # get the current frame from the video stream as an image
    ret,current_frame = webcam_video_stream.read()
    # resize the current frame to 1/4 size to proces faster
    current_frame_small = cv2.resize(current_frame, (0,0), fx= 0.25, fy= 0.25)

    # detect all faces in the image
    all_face_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample= 1, model='hog')
    
    # detect face encodings for all the faces detected
    all_face_encodings = face_recognition.face_encodings(current_frame_small,all_face_locations)

    #looping through the face locations and the face embeddings
    for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):

        #find all the matches and get the list of matches , it will return a list of boolean values
        all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding, tolerance= 0.5)
        
        # calculate and print the face distances
        minimum_distance, percent_of_match = faceDistanceCalculator(known_face_encodings, current_face_encoding)
        print(f"Face distance is {minimum_distance:.2f}")
        print(f"Percentage of match is {percent_of_match:.2f} %")
        
        # initialize with unknown face
        name_of_person = 'Unknown'
        if True in all_matches:
            first_match_index = all_matches.index(True)             # grab the index of match image
            name_of_person = known_names[first_match_index]         # grab the name of corrosponding index
            # update attendance
            markAttendance(name_of_person)


        # we can also use  concept of face distance and k-NN  algorithm for recognising faces    

        #splitting the tuple to get the four position values of current face in clock wise

        #change the position maginitude to fit the actual size video frame
        current_face_location = np.array(current_face_location) * 4

        # grab top, right, botton and left position
        top_pos, right_pos, bottom_pos, left_pos = current_face_location

        #draw rectangle around the face    
        cv2.rectangle(current_frame, (left_pos,top_pos), (right_pos,bottom_pos), (255,0,0),2)

        # filled with color at bottom of face
        cv2.rectangle(current_frame, (left_pos, bottom_pos), (right_pos, bottom_pos+20), (255, 0, 0), cv2.FILLED)

        # display the name as text in the image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, name_of_person, (left_pos,bottom_pos+16), font, 0.5, (255,255,255),1)

    #display the video , outside the for loop
    cv2.imshow("Webcam Video",current_frame)
    
    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#release the stream and cam
#close all opencv windows open
webcam_video_stream.release()
cv2.destroyAllWindows() 
