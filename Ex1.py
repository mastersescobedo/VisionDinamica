
import cv2
import argparse
import numpy as np

if __name__ == "__main__":

    f_path = '1-11200.avi'
    cap = cv2.VideoCapture(f_path)
    alpha = 0.05
    start = 1
    th = 25

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame_act = np.copy(gray)
            if start==1:
                fondo = np.copy(frame_act)
                frame_ant = np.copy(frame_act)
                mascara=np.copy(frame_act)
                start=2
            elif start == 2:

                fondo = alpha * np.float32(frame_ant) + (1-alpha)*np.float32(fondo)

                # for x in range(fondo.shape[0]):
                #     for y in range(fondo.shape[1]):
                #         if fondo[x][y]==frame_ant[x][y]:
                #             fondo[x][y] = alpha * np.float32(frame_ant[x][y]) + (1 - alpha) * np.float32(fondo[x][y])

                # actualizacion = alpha * np.float32(frame_ant) + (1-alpha)*np.float32(fondo)
                #
                # act0 = np.multiply(mascara, fondo)
                # mask_neg = cv2.bitwise_not(mascara)
                #
                # act1 = np.multiply(mask_neg,actualizacion)
                #
                # fondo = (act1 + act0)

                mascara = np.abs(np.float32(frame_act) - np.float32(fondo))

                mascara[mascara>th] = 255
                mascara[mascara!=255]=0

                frame_ant = frame_act
                cv2.imshow('fondo', np.uint8(fondo))
                cv2.imshow('mascara',np.uint8(mascara))

                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
            else:

                actualizacion = alpha * np.float32(frame_ant) + (1-alpha)*np.float32(fondo)

                act0 = np.multiply(mascara, fondo)
                mask_neg = cv2.bitwise_not(mascara)

                act1 = np.multiply(mask_neg,actualizacion)

                fondo = (act1 + act0)

                mascara = np.abs(np.float32(frame_act) - np.float32(fondo))

                mascara[mascara > th] = 255
                mascara[mascara != 255] = 0

                frame_ant = frame_act
                cv2.imshow('fondo', np.uint8(fondo))
                cv2.imshow('mascara', np.uint8(mascara))


                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
        else:
            break
    cap.release()
    # out.release()
    cv2.destroyAllWindows()