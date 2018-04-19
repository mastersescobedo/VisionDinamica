import numpy as np
import cv2
import random
import math

def createMask(low, high, image):
    lower_color = np.array(low, dtype='uint8')
    upper_color = np.array(high, dtype='uint8')
    return cv2.inRange(image, lower_color, upper_color)


if __name__ == "__main__":

    f_path = 'input/pelota.avi'
    cap = cv2.VideoCapture(f_path)

    num_semillas = 100

    item_size = 100
    image_size=(638,360)

    lower_color=[0,140,40]
    upper_color=[10,255,160]

    estados =np.zeros((2,num_semillas))

    estados[0, :] = np.random.random_integers(0, image_size[0] - item_size, size=num_semillas)
    estados[1, :] = np.random.random_integers(0, image_size[1] - item_size, size=num_semillas)

    pesos = np.zeros(num_semillas)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:

            image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            mask = createMask(lower_color, upper_color, image_hsv)
            rectangle=[None]*num_semillas

            for i in range(0,num_semillas):
                rectangle = mask[int(estados[0,i]):int(estados[0,i])+item_size,int(estados[1,i]):int(estados[1,i])+item_size]

                pesos[i]=np.count_nonzero(rectangle)

            #print(pesos)

            suma = pesos.sum(0)

            if suma!=0:
                pesos = pesos / pesos.sum(0)

                #print(pesos)

                acum = np.cumsum(pesos,axis=0)

                print(acum)

                #prob = np.random.rand(10,1)

                #Recorrer para quedarme solo con el primero

                num_rand = 100
                valores_aleatorios = np.zeros((num_rand,1))

                for n_states in range(num_rand):
                    prob = np.random.rand()
                    for ind_acc in range(0,acum.shape[0]):
                        if acum[ind_acc]>=prob:
                            valores_aleatorios[n_states]=ind_acc
                            break



                print("_________________")
                print(valores_aleatorios)
                print("_________________")

                np.random.normal(0,1)

                print(estados[0,:])
                print("______******______")
                print(estados[1, :])
                print("______******______")

                # for new_states in range(valores_aleatorios.shape[0]):
                #
                #     estados[0, new_states] = estados[0, int(valores_aleatorios[new_states])]
                #     estados[1, new_states] = estados[1, int(valores_aleatorios[new_states])]

                # for new_states in range(valores_aleatorios.shape[0]):
                #
                #     state_value_0 = estados[0,int(valores_aleatorios[new_states])]
                #
                #     if state_value_0<1:
                #         estados[0, new_states] = math.fabs(np.random.normal(1, 0.5)*1)
                #     else:
                #         estados[0, new_states] = math.fabs(
                #             np.random.normal(1, 1) * estados[0, int(valores_aleatorios[new_states])])
                #
                #     state_value_1 = estados[1, int(valores_aleatorios[new_states])]
                #
                #     if state_value_1 <1:
                #         estados[1, new_states] = math.fabs(np.random.normal(1, 0.5) * 1)
                #     else:
                #         estados[1, new_states] = math.fabs(
                #             np.random.normal(1, 1) * estados[1, int(valores_aleatorios[new_states])])

                for new_states in range(valores_aleatorios.shape[0]):

                    state_value_0 = estados[0,int(valores_aleatorios[new_states])]

                    if state_value_0>0:
                        estados[0, new_states] = math.fabs(
                            np.random.normal(0, 15) + estados[0, int(valores_aleatorios[new_states])])

                    state_value_1 = estados[1, int(valores_aleatorios[new_states])]

                    if state_value_1 >0:
                        estados[1, new_states] = math.fabs(
                            np.random.normal(0, 15) + estados[1, int(valores_aleatorios[new_states])])



                    #np.random.random_integers(0, image_size[0] - item_size, size=num_semillas)
                    #np.random.random_integers(0, image_size[1] - item_size, size=num_semillas)


                print(estados[0,:])
                print("_____------_______")
                print(estados[1, :])
                print("_____------_______")

            #Se lanza un aleatorios hasat que se encuentra un peso que es mayor que ese numero lanzado



            #usar np.random.rand para generar ya que es uniforme, miro en el acumulado donde esta el valor y luego

            #REcorrer hasta que que enceuntras el elemento con el where


            #ampliar con randn y una vez calculados los valores se perturban los originales

            #j = np.unravel_index(pesos.argmax(), pesos.shape)

            for itera in range(num_semillas):

                end_point_a = int(estados[1, itera])+100
                if end_point_a>image_size[0]:
                    end_point_a=image_size[0]

                end_point_b = int(estados[0, itera])+100
                if end_point_b>image_size[1]:
                    end_point_b=image_size[1]

                out = cv2.rectangle(frame, (int(estados[1, itera]), int(estados[0, itera])), (end_point_a, end_point_b), (0, 255, 0), 3)

            cv2.imshow('1', out)
            cv2.waitKey(100)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break
    cap.release()
    # out.release()
    cv2.destroyAllWindows()
