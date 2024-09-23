from ultralytics import YOLO
import cv2
import numpy as np
import serial
import time 

# Leemos el modelo
model = YOLO('Test.pt')  # Cargamos el modelo YOLOv8

# Color Verde
verded = np.array([40, 80, 80])
verdeu = np.array([80, 220, 220])

# Configuramos el puerto serial
com = serial.Serial("COM4", 9600, write_timeout=10)

# Función para enviar señal y mover el servomotor
def mover_servomotor(senal):
    com.write(senal.encode('ascii'))
    time.sleep(1)  # Esperamos un poco antes de cerrar el puerto serial

# Función para abrir la puerta
def abrir_puerta():
    mover_servomotor('a')

# Función para cerrar la puerta
def cerrar_puerta():
    mover_servomotor('c')

# Realizo Videocaptura
cap = cv2.VideoCapture(1)

# Variables
contafot = 0
contacar = 0
marca = 0
flag1 = 0
flag2 = 0

# Empezamos
while True:
    # Realizamos lectura de frames
    ret, frame = cap.read()

    # Creamos copia
    copia = frame.copy()

    # Mostramos el número de vehículos
    cv2.putText(frame, "Ocupacion: ", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, str(contacar), (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, "Carros", (240, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Mejoramos rendimiento
    contafot += 1
    if contafot % 3 != 0:
        continue

    # Realizamos las detecciones
    results = model(frame)  # Usamos el modelo YOLOv8 para obtener los resultados

    # Extraemos la info
    info = results[0].boxes.xyxy.numpy()  # Extraemos las predicciones de las cajas delimitadoras

    # Preguntamos si hay detecciones
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]

        label = results[0].names[int(box.cls[0])]
        confidence = box.conf[0]

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convertimos la copia a HSV para buscar la marca en el suelo
        hsv = cv2.cvtColor(copia, cv2.COLOR_BGR2HSV)

        # Creamos la máscara para detectar el color verde
        mask = cv2.inRange(hsv, verded, verdeu)

        # Encontramos contornos
        contornos, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contornos = sorted(contornos, key=lambda x: cv2.contourArea(x), reverse=True)

        # Detectamos la marca
        for ctn in contornos:
            # Extraemos información de la marca
            xiz, yiz, ancho, alto = cv2.boundingRect(ctn)
            xfz, yfz = ancho + xiz, alto + yiz
            cv2.rectangle(frame, (xiz, yiz), (xfz, yfz), (0, 255, 0), 2)

            # Extraemos el centro
            cxm, cym = (xiz + xfz) // 2, (yiz + yfz) // 2
            cv2.circle(frame, (cxm, cym), 2, (0, 255, 0), 3)

            # Delimitamos zonas de interés
            linxe = cxm + 70  # Entrada
            linxs = cxm - 70  # Salida

            cv2.line(frame, (linxe, yiz), (linxe, yfz), (0, 0, 255), 2)
            cv2.line(frame, (linxs, yiz), (linxs, yfz), (0, 0, 255), 2)
            cv2.circle(frame, (20, 20), 15, (0, 0, 255), cv2.FILLED)

            # Si el carro está en la zona de entrada
            if x1 < linxe < x2 and flag1 == 0 and flag2 == 0 or marca == 1:
                print("ENTRADA")
                flag1 = 1

                cv2.circle(frame, (20, 20), 15, (0, 255, 255), cv2.FILLED)
                cv2.line(frame, (linxe, yiz), (linxe, yfz), (0, 255, 255), 2)

                abrir_puerta()

                marca = 1
                if x1 < linxs < x2 and flag1 == 1:
                    print("ENTRADA2")
                    cv2.putText(frame, "ENTRANDO", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.line(frame, (linxs, yiz), (linxs, yfz), (0, 255, 255), 2)

                    flag2 = 1

                elif x2 < linxs and flag2 == 1:
                    print("ENTRADA3")
                    cerrar_puerta()
                    marca = 0
                    flag1 = 0
                    flag2 = 0
                    contacar += 1

            # Si el carro está en la zona de salida
            elif x1 < linxs < x2 and flag1 == 0 and flag2 == 0 or marca == 2:
                print("SALIDA")
                flag2 = 2

                cv2.circle(frame, (20, 20), 15, (0, 255, 255), cv2.FILLED)
                cv2.line(frame, (linxs, yiz), (linxs, yfz), (0, 255, 255), 2)

                abrir_puerta()
                marca = 2

                if x1 < linxe < x2 and flag2 == 2:
                    print("SALIDA2")
                    cv2.putText(frame, "SALIENDO", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.line(frame, (linxe, yiz), (linxe, yfz), (0, 255, 255), 2)

                    flag1 = 2

                elif x1 > linxe and flag1 == 2:
                    print("SALIDA3")
                    cerrar_puerta()
                    marca = 0
                    flag1 = 0
                    flag2 = 0
                    contacar -= 1

            break

    # Mostramos el frame
    cv2.imshow('Parqueadero', frame)

    # Leemos el teclado
    t = cv2.waitKey(5)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()
