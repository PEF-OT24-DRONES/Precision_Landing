import numpy as np
import cv2
import cv2.aruco as aruco
import sys, time, math
import json
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

#------------------------------------------------------------------------------
#------- ROTATIONS https://www.learnopencv.com/rotation-matrix-to-euler-angles/
#------------------------------------------------------------------------------
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

#--- Define Tag
id_to_find  = 100
marker_size  = 18.5  # [cm]

#--- Calibration Path
calib_path  = "/home/albertocastro/Documents/Precision_Landing/scripts/"
camera_matrix   = np.loadtxt(calib_path+'cameraMatrix_drone.txt', delimiter=',')
camera_distortion   = np.loadtxt(calib_path+'cameraDistortion_drone.txt', delimiter=',')

#--- 180 deg rotation matrix around the x axis
R_flip  = np.array([[1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]], dtype=np.float32)

#--- ArUco Dictionary
aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
parameters  = aruco.DetectorParameters()

#--- Capture the video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#--- Font
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar los marcadores ArUco
    corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters)

    if ids is not None and id_to_find in ids:
        index = np.where(ids == id_to_find)[0][0]
        ret = aruco.estimatePoseSingleMarkers([corners[index]], marker_size, camera_matrix, camera_distortion)

        rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]

        # Dibujar el marcador detectado y un círculo en el centro
        aruco.drawDetectedMarkers(frame, corners)
        center = tuple(corners[index][0].mean(axis=0).astype(int))
        cv2.circle(frame, center, 10, (0, 255, 0), -1)  # Círculo verde en el centro del marcador

        # Dibujar ejes del sistema de coordenadas de la cámara
        axis_length = 10
        axis = np.float32([[axis_length,0,0], [0,axis_length,0], [0,0,axis_length]]).reshape(-1,3)
        imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_distortion)

        imgpts = np.int32(imgpts).reshape(-1, 2)
        frame = cv2.line(frame, center, tuple(imgpts[0]), (255, 0, 0), 5)  # Eje X en rojo
        frame = cv2.line(frame, center, tuple(imgpts[1]), (0, 255, 0), 5)  # Eje Y en verde
        frame = cv2.line(frame, center, tuple(imgpts[2]), (0, 0, 255), 5)  # Eje Z en azul

        # Imprimir la posición del marcador en la terminal
        print(f"MARKER Position x={tvec[0]:.2f} cm, y={tvec[1]:.2f} cm, z={tvec[2]:.2f} cm")

        # Obtener la matriz de rotación del marcador respecto a la cámara
        R_ct = np.matrix(cv2.Rodrigues(rvec)[0])
        R_tc = R_ct.T

        # Convertir la rotación a ángulos de Euler
        roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip * R_tc)

        # Imprimir la actitud del marcador en la terminal
        print(f"MARKER Attitude roll={math.degrees(roll_marker):.2f} deg, pitch={math.degrees(pitch_marker):.2f} deg, yaw={math.degrees(yaw_marker):.2f} deg")

        # Calcular la posición de la cámara respecto al marcador
        pos_camera = -R_tc * np.matrix(tvec).T
        print(f"CAMERA Position x={pos_camera[0, 0]:.2f} cm, y={pos_camera[1, 0]:.2f} cm, z={pos_camera[2, 0]:.2f} cm")

        # Calcular la actitud de la cámara respecto al marco
        roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(R_flip * R_tc)
        print(f"CAMERA Attitude roll={math.degrees(roll_camera):.2f} deg, pitch={math.degrees(pitch_camera):.2f} deg, yaw={math.degrees(yaw_camera):.2f} deg")

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
