import cv2
import numpy as np
import math
from time import time

# No hacers saltos ne als ventanas de integracion
# Se calcularan las flechas en cada pixel, entonces agrupar
# tiempo real 1/20

if __name__ == "__main__":

    f_path = 'cama.avi'
    cap = cv2.VideoCapture(f_path)

    # Inicio del calculo del tiempo
    inicio = time()

    # Tamaño del kernel y valor para los movimientos de bucle
    kernel = 5
    k = math.floor(kernel / 2)

    # Inicialización de A, B y uv
    A = np.zeros((2, 2))
    B = np.zeros((2, 1))
    uv = np.zeros((2, 1))

    start = True

    # small = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:

            if start:
                imagen_anterior = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255, (0, 0), fx=0.5, fy=0.5)
                # imagen_anterior = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255

                start = False

            else:

                inicio = time()

                imagen_actual = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255, (0, 0), fx=0.5, fy=0.5)
                # imagen_actual = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255

                h, w = imagen_anterior.shape

                # Inicialización de la imagen de salida como la imagen de la siguiente iteración
                imagen_out = np.copy(imagen_actual)

                # Filtrado de las imagenes para crear It
                imagen1_1 = cv2.GaussianBlur(imagen_anterior, (5, 5), 0.2)
                imagen2_2 = cv2.GaussianBlur(imagen_actual, (5, 5), 0.2)

                # Creación de las derivadas parciales  Ix e Iy, realizando la media entre la actual y la siguiente
                derx1 = cv2.Sobel(imagen_anterior, cv2.CV_64F, 1, 0, 3)
                dery1 = cv2.Sobel(imagen_anterior, cv2.CV_64F, 0, 1, 3)

                derx2 = cv2.Sobel(imagen_actual, cv2.CV_64F, 1, 0, 3)
                dery2 = cv2.Sobel(imagen_actual, cv2.CV_64F, 0, 1, 3)

                derx = (derx1[:, :] + derx2[:, :]) / 2
                dery = (dery1[:, :] + dery2[:, :]) / 2

                # Creación de la derivada temporal  It
                dert = imagen2_2 - imagen1_1

                # Multiplicaciones de la fórmula 1 de T-K
                Ix2_2 = np.multiply(derx, derx)
                Iy2_2 = np.multiply(dery, dery)
                Ixy_2 = np.multiply(derx, dery)
                Ixt_2 = np.multiply(derx, dert)
                Iyt_2 = np.multiply(dery, dert)

                # Variables de tamaño para controlar los bucles
                hk = h  # int(np.ceil(h / (kernel) - 1))
                wk = w  # int(np.ceil(w / (kernel) - 1))

                # Se recorre la imagen para empezar a hacer los cálculos de la fórmula 1
                for i in range(hk, 0, -1):
                    for j in range(wk, 0, -1):
                        # Variables para creación de los kernels que recorreran la imagen
                        ii = i  # * kernel + k
                        jj = j  # * kernel + k

                        # Sumatorios de la fórmula 2
                        Ix2 = np.sum(Ix2_2[ii - k:ii + k, jj - k:jj + k])
                        Iy2 = np.sum(Iy2_2[ii - k:ii + k, jj - k:jj + k])
                        Ixy = np.sum(Ixy_2[ii - k:ii + k, jj - k:jj + k])
                        Ixt = np.sum(Ixt_2[ii - k:ii + k, jj - k:jj + k])
                        Iyt = np.sum(Iyt_2[ii - k:ii + k, jj - k:jj + k])

                        # Cálculo de los diferentes productos de los sumatorios anteriores
                        U1 = Iy2 * Ixt * (-1)
                        U2 = Ixy * Iyt
                        U3 = Ix2 * Iy2
                        U4 = Ixy * Ixy
                        U5 = Ixy * Ixt
                        U6 = Ix2 * Iyt
                        U8 = Ixy * Ixy

                        # Cälculo de u y v
                        u = (U1 + U2) / (U3 - U4)
                        v = (U5 - U6) / (U3 - U8)

                        if (U3 - U4) == 0:
                            u = 0
                        if (U3 - U8) == 0:
                            v = 0

                        if v > 255:
                            v = 255
                        if u > 255:
                            u = 255

                        if v < -255:
                            v = -255
                        if u < -255:
                            u = -255

                        if ii % kernel == 0 and jj % kernel == 0:
                            # Dibujo en la imagen de salida de los vectores de velocidad obtenidos
                            cv2.arrowedLine(imagen_out, (jj, ii), (jj + int(u), ii + int(v)), (255, 255, 255))

                # Mostrar y guardar la imagen de salida para su posterior análisis
                cv2.imshow('1', imagen_out)

                cv2.waitKey(1)
                # imagen = imagenes[x].replace('.jpg', '')
                # print(imagen)
                # cv2.imwrite(imagen + '_pro' + '.jpg', imagen_out)

                imagen_anterior = imagen_actual

            # Cálculo del tiempo del script y mostrarlo por pantalla
            tiempototal = time() - inicio
            print('LK1 ha tardado = ' + str(tiempototal))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    # out.release()
    cv2.destroyAllWindows()
