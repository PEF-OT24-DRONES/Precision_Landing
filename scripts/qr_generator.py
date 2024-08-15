import qrcode

# Información que deseas codificar en el QR Code
data = {
    "landing_zone": "Zone_A",
    "coordinates": {
        "latitude": 25.686614,
        "longitude": -100.316113,
        "altitude": 10.0
    },
    "orientation": 90  # Orientación en grados
}

# Convertir la información a formato JSON (opcional)
import json
qr_data = json.dumps(data)

# Generar el QR Code
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data(qr_data)
qr.make(fit=True)

# Crear la imagen del QR Code
img = qr.make_image(fill_color="black", back_color="white")

# Guardar la imagen en un archivo
img.save("drone_landing_qr.png")
