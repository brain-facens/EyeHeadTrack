#!/home/nata-brain/miniconda3/envs/eyegaze/bin/python3

import rospy 
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

cap = cv2.VideoCapture(0)
print(cap.isOpened())
bridge = CvBridge

if not cap.isOpened():
    print("Camera is not working!")
    
def talker():    
    pub = rospy.Publisher('/webcam', Image, queue_size = 1)
    rospy.init_node('image', anonymous = False)
    rate = rospy.Rate(10)
    
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        
        #print(type(frame) == np.ndarray)
        if type(frame) == np.ndarray:
            msg = bridge.cv2_to_imgmsg(frame, "bgr8")
            pub.publish(msg) 
            
        if cv2.waitKey(1) == ord('q'):
            break
        
        if rospy.is_shutdown():
            cap.release()
    
if __name__ == "__main__":
    try:
        talker()
    except rospy.ROSInterruptException:
        pass