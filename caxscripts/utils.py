
import time
import numpy as np
from siriuspy.devices import CAXCtrl, DVF


CAX = CAXCtrl()


MAXERRORCOUNT = 5

def get_image(dvf: DVF):

    count = 0
    while count < MAXERRORCOUNT:
        try:
            if not dvf.acquisition_status:
                dvf.cmd_acquire_on()
            return dvf.image
        except Exception as err:
            print(f" WARNING. When trying to fetch image from DVF1: {err} ")
            time.sleep(2)
            count += 1
            if count < MAXERRORCOUNT:
                print("\n Repeating the procedure...\n")
            else:
                raise Exception("Client exception")

