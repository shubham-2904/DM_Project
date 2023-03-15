import os
import pprint

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pymongo
from pprint import pprint

import util

# function to find the ev and non ev vehicle
def countingCar():
    # define constants

    model_cfg_path = os.path.join('.', 'model', 'cfg', 'darknet-yolov3.cfg')
    model_weights_path = os.path.join('.', 'model', 'weights', 'model.weights')
    class_names_path = os.path.join('.', 'model', 'classes.names')

    # variable to count ev and non_ev
    ev = 0
    non_ev = 0

    folder_number = int(input("Select folder number number: \n1. data1\n2. data2\n3. data3\n4. data4\n"))

    # getting input images from dir 
    input_dir = f'F:/Ramdeobaba/sem_3/Data Minning Project/Projects/data{folder_number}'

    for image_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, image_name)

        # load class names
        with open(class_names_path, 'r') as f:
            class_names = [j[:-1] for j in f.readlines() if len(j) > 2]
            f.close()

        # load model
        net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)

        # load image

        img = cv2.imread(img_path)

        H, W, _ = img.shape

        # convert image
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)

        # get detections
        net.setInput(blob)

        detections = util.get_outputs(net)

        # bboxes, class_ids, confidences
        bboxes = []
        class_ids = []
        scores = []

        for detection in detections:
            # [x1, x2, x3, x4, x5, x6, ..., x85]
            bbox = detection[:4]

            xc, yc, w, h = bbox
            bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

            bbox_confidence = detection[4]

            class_id = np.argmax(detection[5:])
            score = np.amax(detection[5:])

            bboxes.append(bbox)
            class_ids.append(class_id)
            scores.append(score)

        # apply nms
        bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)

        # plot

        for bbox_, bbox in enumerate(bboxes):
            xc, yc, w, h = bbox
            licensePlate = img[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)):int(xc + (w / 2)), : ].copy()

            img = cv2.rectangle(img,
                                (int(xc - (w / 2)), int(yc - (h / 2))),
                                (int(xc + (w / 2)), int(yc + (h / 2))),
                                (0, 255, 0),
                                10)

        plt.imshow(cv2.cvtColor(licensePlate, cv2.COLOR_BGR2RGB))

        img = cv2.cvtColor(licensePlate, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # print(hsv)

        mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))

        ## slice the green
        imask = mask > 0
        green = np.zeros_like(img, np.uint8)
        green[imask] = img[imask]

        green = 0
        nonGreen = 0

        for i in imask:
            for j in i:
                if j :
                    green += 1
                else:
                    nonGreen += 1

        print(f'Green: {green} and Non green: {nonGreen}')
        if green > nonGreen or abs(green - nonGreen) < 500:
            print("True")
            ev += 1
        else:
            print("False")
            non_ev += 1

        print()

    return ev, non_ev


# data update in existing camera
def updateData(dataFromDatabase, camera):
    oldData = dataFromDatabase["square_data"]
    day_number = int(input("Select Day:\n1. Monday\n2. Tuesday\n3. Wednesday\n4. Thursday\n5. Friday\n6. Saturday\n7. Sunday\n"))
    weeks = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    day = weeks[day_number - 1]

    i = 0

    for index in oldData:
        if index["camera_number"] == camera:
            break

        i += 1

    ev, non_ev = countingCar()

    newDataForEv = {
        "day" : day,
        "count" : ev
    }

    newDataForNonEv = {
        "day" : day,
        "count" : non_ev
    }

    oldData[i]['ev'].append(newDataForEv)
    oldData[i]['non_ev'].append(newDataForNonEv)

    r = collection.update_one(
        {
            "_id": dataFromDatabase["_id"]
        },
        {
            "$set": {
                "square_data" : oldData
            }
        }
    )

    print(r)


def updateDataNewCamera(dataFromDatabase, camera):
    """
    This function is to add data of new camera in existing square
    """
    day_number = int(input("Select Day:\n1. Monday\n2. Tuesday\n3. Wednesday\n4. Thursday\n5. Friday\n6. Saturday\n7. Sunday\n"))
    weeks = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    day = weeks[day_number - 1]

    squareDate = dataFromDatabase["square_data"]
    ev, non_ev = countingCar()



    data = {
        "camera_number": camera,
        "ev" :[{
            "day" : day,
            "count": ev
        }],
        "non_ev": [{
            "day": day,
            "count" : non_ev
        }]
    }

    squareDate.append(data)

    print(squareDate)

    r = collection.update_one(
        {
            "_id": dataFromDatabase["_id"]
        },
        {
            "$set": {
                "square_data" : squareDate
            }
        }
    )
    print(r)


# Main Method
# Database connection setup
myclient = pymongo.MongoClient("mongodb://localhost:27017/") # database client

database = myclient["dm_project"]   # database
collection = database["data"]       # collection
# Database connection setup end

while True:
    square_name = input("Enter square name: ")

    if collection.count_documents({"square_name": square_name}) == 0:
        # print("Data not present")
        connectedSquare = input("Enter connected square: ")
        connected_square = []
        connected_square = connectedSquare.split(" ")

        day_number = int(input("Select Day:\n1. Monday\n2. Tuesday\n3. Wednesday\n4. Thursday\n5. Friday\n6. Saturday\n7. Sunday\n"))
        weeks = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]

        day = weeks[day_number - 1]

        ev, non_ev = countingCar()

        # creating doc for new square data
        newData = {
            "square_name" : square_name,
            "connected_square" : connected_square,
            "square_data" : [
                {
                    "camera_number" : 1,
                    "ev" : [
                        {
                            "day" : day,
                            "count" : ev
                        
                        }
                    ],
                    "non_ev" : [
                        {
                            "day" : day,
                            "count" : non_ev
                        }
                    ]
                }
            ]
        }

        result = collection.insert_one(newData)
        print(result)

    else:
        # getting data from database
        dataFromDatabase = collection.find_one({"square_name": square_name})
        # camera number in which you want to append or eneter data
        camera = int(input("Enter camera number:"))

        
        findCamera = False

        for d in dataFromDatabase['square_data']:
            if d['camera_number'] == camera:
                findCamera = True
                break

        # checking weather the cameara is exist or not
        if findCamera:
            updateData(dataFromDatabase, camera)
        else:
            updateDataNewCamera(dataFromDatabase, camera)
        
    ans = int(input("Do you want to continue:\n1. True\n2. False\n"))
    if ans == 2:
        break
 
