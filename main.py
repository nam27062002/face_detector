from extract import FrameExtractor
from face_detector import FaceExtractor
from model import ImageClassifier
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if "__main__" == __name__:

    # extract video
    start_time = time.time()
    video_path = 'video.mp4'
    output_folder = 'Assets/Extract'
    second_interval = 1
    extractor = FrameExtractor(video_path, output_folder, second_interval)
    extractor.extract_frames()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Thời gian thực hiện extract video: {execution_time} giây")

    # face detector
    start_time = time.time()
    face_extractor = FaceExtractor()
    face_extractor.process_images()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Thời gian thực hiện face detector: {execution_time} giây")

    # matching
    start_time = time.time()
    directory = 'Assets/Cache/'
    classifier = ImageClassifier("keras_Model.h5", "labels.txt")
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    ans = []
    for file in files:
        class_name, _ = classifier.predict(f"Assets/Cache/{file}")
        print("Class:", class_name)
        if class_name:
            ans.append(file)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Thời gian thực hiện matching: {execution_time} giây")

    print(ans)



