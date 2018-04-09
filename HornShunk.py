import cv2
import numpy as np
import numpy.linalg as nl
import math
from time import time

'''
Iteracion para buscar la convergencia, dependientdo del numero de iteraciones será de mayor o menor tiempo de computo

ecuacion diapo 31

Se realiza el bucle sobre la imagen 

donde se suman los valores por el kernel

Se decien pintar cierto pixeles
'''
if __name__ == "__main__":

    f_path = 'escaleras.avi'
    cap = cv2.VideoCapture(f_path)

    # Tamaño del kernel y valor para los movimientos de bucle
    kernel = 5
    k = math.floor(kernel / 2)

    # Inicialización de kernel para suavizado
    kmean = np.ones((kernel, kernel))

    # Inicialización de los parámetros Iteraciones y lambda
    itera = 50
    lam = 3
    lam = lam * lam

    start = True

    # small = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:

            if start:
                #imagen_anterior = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255,(0,0), fx=0.5, fy=0.5)
                imagen_anterior = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255

                start = False
            else:

                # Inicio del calculo del tiempo
                inicio = time()

                #imagen_actual = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255,(0,0), fx=0.5, fy=0.5)
                imagen_actual = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255

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

                # Creación de las variables umean y vmean
                umean = np.zeros((h, w))
                vmean = np.zeros((h, w))


                # Bucle que se encargue de realizar las iteraciones que se programen
                for iter in range(itera):
                    # Filtrado de suavizado para umean y vmena
                    umean = cv2.filter2D(umean, cv2.CV_64F, kmean) / (kernel * kernel)
                    vmean = cv2.filter2D(umean, cv2.CV_64F, kmean) / (kernel * kernel)

                    # Cáculo de la nueva umean y vmean
                    div = (np.multiply(derx, umean) + np.multiply(dery, vmean) + dert) / (
                        lam + derx2[:, :] + dery2[:, :])
                    umean = umean - (derx * div)
                    vmean = vmean - (dery * div)

                # Variables de tamaño para controlar los bucles
                hk = h  # int(np.ceil(h / (kernel) - 1))
                wk = w  # int(np.ceil(w / (kernel) - 1))

                # Inicialización de la variable u y v
                u = np.zeros((hk, wk))
                v = np.zeros((hk, wk))

                # Se recorre la imagen para dibujar los vectores u y v de H&S
                for i in range(hk, k, -1):
                    for j in range(wk, 0, -1):
                        # Variables para creación de los kernels que recorreran la imagen
                        ii = i  # * kernel + k
                        jj = j  # * kernel + k

                        # Sumatorios de los valores englobados en el kernel definido
                        u = sum(sum(umean[ii - k:ii + k, jj - k:jj + k]))
                        v = sum(sum(vmean[ii - k:ii + k, jj - k:jj + k]))

                        # Transformación a enteros para su posterior dibujado
                        u0 = int(u)
                        v0 = int(v)

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
