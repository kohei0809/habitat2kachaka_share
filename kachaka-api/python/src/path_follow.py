import sys
import os

import numpy as np
from utils import move

sys.path.append(f"/home/{os.environ['USER']}/Desktop/habitat2kachaka/kachaka-api/python/")
import kachaka_api


if __name__ == "__main__":
    client = kachaka_api.KachakaApiClient(sys.argv[1])
    print(client.get_locations())
    # 棚をとりにいく
    #client.move_shelf("S01", "start")
    #move(client, 3.8, 1.5, np.pi)
    
    while True:
        move(client, 3.8, 1.5, np.pi)
        move(client, 3.8, -2.5, np.pi / 2)
    
