import cv2
from imutils.video import VideoStream

thres = 0.5                              # Threshold to detect objects


# img = cv2.imread('Shubham.jpg')
# cap = cv2.VideoCapture(0)              # Using cv2
cap = VideoStream(src=0).start()
# cap = VideoStream(src="http://192.168.0.102:8080/video").start()
# cap.set(3, 640)
# cap.set(4, 480)

classNames = []
classFile = 'objects.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


while True:
    img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    # print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
            cv2.putText(img, classNames[classId-1].upper(), (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.putText(img, str(confidence), (box[0]+30, box[1] + 50), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (255, 0, 0), 2)


    cv2.imshow('[TERMINATORS] Drone cam Stream', img)
    cv2.waitKey(0)
