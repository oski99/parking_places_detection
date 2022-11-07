import cv2
import deeplake
import numpy
import matplotlib.pyplot as plt
import random

from parking_places_detector import ParkingPlacesDetector

if __name__ == "__main__":
    ds = deeplake.load('hub://activeloop/carpk-train')

    detector = ParkingPlacesDetector()

    iter_list = list(range(1, len(ds.images)))
    random.shuffle(iter_list)

    for i in iter_list[:5]:
        img = ds.images[i].numpy()
        label = ds.labels[i].numpy()

        img_det, count = detector.detect(img)
        log = f'image id: {i}, cars: {len(label)}, detected: {count}'
        print(log)
        plt.imshow(img_det)
        plt.title(log)

        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()
