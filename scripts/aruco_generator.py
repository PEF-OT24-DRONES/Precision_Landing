import cv2
import cv2.aruco as aruco

# Selecciona el diccionario de ArUco (puedes elegir otro si lo prefieres)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)

# Define el ID del marcador que quieres generar
marker_id = 100

# Tamaño del marcador (en pixeles)
marker_size = 700

# Genera el marcador
marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

# Guarda el marcador en una imagen
cv2.imwrite(f'aruco_marker_{marker_id}.png', marker_image)

# Mostrar el marcador generado
cv2.imshow('Aruco Marker', marker_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

