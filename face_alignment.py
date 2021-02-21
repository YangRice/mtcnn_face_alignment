import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

INPUT_IMAGE = 'ziyu.jpg'
OUTPUT_IMAGE = 'output.png'
detector = MTCNN(steps_threshold=[0.0, 0.0, 0.0])

def landmarks(img):
    faces = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    face = max(faces, key=lambda x: x['confidence'])
    return face['keypoints']

def affineMatrix(lmks, scale=2.5):
    nose = np.array(lmks['nose'], dtype=np.float32)
    left_eye = np.array(lmks['left_eye'], dtype=np.float32)
    right_eye = np.array(lmks['right_eye'], dtype=np.float32)
    eye_width = right_eye - left_eye
    angle = np.arctan2(eye_width[1], eye_width[0])
    center = nose
    alpha = np.cos(angle)
    beta = np.sin(angle)
    w = np.sqrt(np.sum(eye_width**2)) * scale
    m = [[alpha, beta, -alpha * center[0] - beta * center[1] + w * 0.5],
        [-beta, alpha, beta * center[0] - alpha * center[1] + w * 0.5]]
    return np.array(m), (int(w), int(w))

if __name__ == '__main__':
    img = cv2.imread(INPUT_IMAGE)
    mat, size = affineMatrix(landmarks(img))
    cv2.imwrite(OUTPUT_IMAGE, cv2.warpAffine(img, mat, size))
