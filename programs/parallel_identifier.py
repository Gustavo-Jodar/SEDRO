import cv2
from mpi4py import MPI
import numpy as np
from p3_function import p3
from p2_function import p2

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if rank == 0:
    FILE = "../video.MP4"
    cap = cv2.VideoCapture(FILE)

    ret, first_frame = cap.read()

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (600, 500))

        if not ret:
            break

        img_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        img_size = len(img_bytes)
        
        for proc in range(1, size): 
            # send data size to other processes
            comm.send(img_size, dest=proc, tag=0)
            # send the image to other processes
            comm.send(img_bytes, dest=proc, tag=1)

        cv2.imshow('process 0', frame)
        
        if cv2.waitKey(1) & (0xFF == ord('q') ):
            break

    cv2.destroyAllWindows()
    cap.release()
    
else:
    while True:
        # rcvs img size
        img_size = comm.recv(source=0, tag=0)
        # rcvs image
        img_bytes = comm.recv(source=0, tag=1)

        # byters -> numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)

        # decode array to imagem
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        #process 1 run programme 2
        if(rank == 1):
            p2(img)
        
        #process 2 run programme 3
        if(rank == 2):
            p3(img)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
