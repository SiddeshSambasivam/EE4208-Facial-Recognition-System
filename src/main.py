import logging
import argparse

import cv2
import numpy as np
from imutils.video import FPS

from data import make_dataset
from models import load_detection_model, fit_model

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.INFO)

def main(args):

    threshold = args.threshold
    dim_size = args.dim_size
    path = args.path
    model_path = args.model_path

    dataset = make_dataset(path)
    model, pca = fit_model(dataset, dim_size)

    src = cv2.VideoCapture(0)
    fps = FPS().start()

    net = load_detection_model(model_path)
    
    size = (int(src.get(cv2.CAP_PROP_FRAME_WIDTH)), int(src.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out_fps = 20  
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    writer = cv2.VideoWriter()
    out_path = model_path +'/out.mp4'    
    writer.open(out_path, fourcc, out_fps, size, True)

    while True:

        _, frame = src.read()
        origin_h, origin_w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300))
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]
            if confidence > threshold:

                bounding_box = detections[0, 0, i, 3:7] * np.array([origin_w, origin_h, origin_w, origin_h])
                x_start, y_start, x_end, y_end = bounding_box.astype('int')                
                
                face = frame[y_start:y_end, x_start:x_end]
                try:
                    gray = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, (300, 300))

                    embedding = pca.transform(gray.flatten().reshape(1, -1))
                    label = model.predict(embedding)

                    id = dataset.encoder.inverse_transform(label)
                    label = str(id[0])  

                    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end),(0, 0, 255), 2)
                    cv2.rectangle(frame, (x_start, y_start-18), (x_end, y_start), (0, 0, 255), -1)
                    cv2.putText(frame, label, (x_start+2, y_start-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                except Exception as e:
                    logger.error(e)

        fps.update()
        fps.stop()
        text = "FPS: {:.2f}".format(fps.fps())

        cv2.putText(frame, text, (15, int(origin_h * 0.92)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('Frame', frame)
        writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"): 
            break
    
    writer.release()
    src.release()
    cv2.destroyAllWindows()    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="Path to the dataset")
    parser.add_argument("-d", "--dim_size", type=int, default=50, help="Number of components")
    parser.add_argument("-m", "--model_path", help="Path to the model")
    parser.add_argument("-t", "--threshold", type=float, default=0.7, help="Threshold")

    args = parser.parse_args()

    main(args)