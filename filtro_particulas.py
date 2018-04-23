import numpy as np
import cv2
import random
import math


def createMask(low, high, image):
    lower_color = np.array(low, dtype='uint8')
    upper_color = np.array(high, dtype='uint8')
    return cv2.inRange(image, lower_color, upper_color)


def inicializacion(num_semillas, item_size):
    # Se crean tantos estados como semillas
    estados = np.zeros((2, num_semillas))

    # Cada estado se compone de dos componentes, el vertice superior izquierdo a partir del cual se crearán los rectanculos de deteccón
    estados[0, :] = np.random.random_integers(0, image_size[1] - item_size, size=num_semillas)
    estados[1, :] = np.random.random_integers(0, image_size[0] - item_size, size=num_semillas)

    # Se definen tantos pesos como semillas
    pesos = np.zeros(num_semillas)

    return estados, pesos


def evaluacion(frame, lower_color, upper_color, num_semillas, item_size):
    # Se convierte el frame a HSV
    image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Se aplia una máscara con el color a detectar
    mask = createMask(lower_color, upper_color, image_hsv)
    rectangle = [None] * num_semillas

    # Se dibuja un rectangulo en la imagen y se cuentan los pixeles con valores positivos para cada rectangulo/estado
    for i in range(0, num_semillas):
        rectangle = mask[int(estados[0, i]):int(estados[0, i]) + item_size,
                    int(estados[1, i]):int(estados[1, i]) + item_size]

        pesos[i] = np.count_nonzero(rectangle)

    # Se realiza la suma de los pesos
    suma = pesos.sum(0)

    return suma, pesos


def estimacion(frame, pesos, estados, item_size):
    index = np.argmax(pesos)

    end_point_a = int(estados[1, index]) + item_size
    if end_point_a > image_size[0]:
        end_point_a = image_size[0]

    end_point_b = int(estados[0, index]) + item_size
    if end_point_b > image_size[1]:
        end_point_b = image_size[1]

    out = cv2.rectangle(frame, (int(estados[1, index]), int(estados[0, index])), (end_point_a, end_point_b),
                        (0, 255, 0), 3)
    return out


def seleccion(pesos, num_semillas):
    # Normalización de los pesos
    pesos = pesos / pesos.sum(0)

    # Calculo del vector acumulado
    acum = np.cumsum(pesos, axis=0)

    print(acum)

    # Se generan tantos numero aleatorios como estados
    valores_aleatorios = np.zeros((num_semillas, 1))

    # Se evalua cada numero aleatorio en el vector de probabilidades acumuladas y se obtienen los estados para generar la nueva poblacion
    for n_states in range(num_semillas):
        prob = np.random.rand()
        for ind_acc in range(0, acum.shape[0]):
            if acum[ind_acc] >= prob:
                valores_aleatorios[n_states] = ind_acc
                break

    return valores_aleatorios


def difusion_original(estados, valores_aleatorios):
    for new_states in range(valores_aleatorios.shape[0]):

        state_value_0 = estados[0, int(valores_aleatorios[new_states])]

        if state_value_0 > 0:
            estados[0, new_states] = math.fabs(
                np.random.normal(0, 15) + estados[0, int(valores_aleatorios[new_states])])

        state_value_1 = estados[1, int(valores_aleatorios[new_states])]

        if state_value_1 > 0:
            estados[1, new_states] = math.fabs(
                np.random.normal(0, 15) + estados[1, int(valores_aleatorios[new_states])])

    return estados


def difusion(estados, valores_aleatorios):
    for new_states in range(valores_aleatorios.shape[0]):
        # Se perturban las componentes del punto que sirven de origen para el recuadro
        estados[0, new_states] = math.fabs(np.random.normal(0, 15) + estados[0, int(valores_aleatorios[new_states])])
        estados[1, new_states] = math.fabs(np.random.normal(0, 15) + estados[1, int(valores_aleatorios[new_states])])

    return estados


if __name__ == "__main__":

    f_path = 'input/pelota.avi'
    cap = cv2.VideoCapture(f_path)

    num_semillas = 100

    item_size = 150
    image_size = (638, 360)

    lower_color = [0, 140, 40]
    upper_color = [10, 255, 160]

    no_visible = 0

    # Definición del los estados aleatorios
    estados, pesos = inicializacion(num_semillas, item_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:

            if no_visible >= 10:
                estados, pesos = inicializacion(num_semillas, item_size)

            # Evaluación de los estados
            suma, pesos = evaluacion(frame, lower_color, upper_color, num_semillas, item_size)

            no_visible += 1

            out = np.copy(frame)

            # Si se ha detectado el objeto en algun estado
            if suma != 0:
                out = estimacion(frame, pesos, estados, item_size)

                valores_aleatorios = seleccion(pesos, num_semillas)

                estados = difusion(estados, valores_aleatorios)

                visible = 0

            # for itera in range(num_semillas):
            #
            #     end_point_a = int(estados[1, itera])+100
            #     if end_point_a>image_size[0]:
            #         end_point_a=image_size[0]
            #
            #     end_point_b = int(estados[0, itera])+100
            #     if end_point_b>image_size[1]:
            #         end_point_b=image_size[1]
            #
            #     out = cv2.rectangle(frame, (int(estados[1, itera]), int(estados[0, itera])), (end_point_a, end_point_b), (0, 255, 0), 3)

            cv2.imshow('1', out)
            cv2.waitKey(22)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break
    cap.release()
    # out.release()
    cv2.destroyAllWindows()
