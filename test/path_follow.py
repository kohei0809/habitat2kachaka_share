import sys
import os

import numpy as np
import kachaka_api
from utils import move
sys.path.append(f"/home/{os.environ['USER']}/Desktop/habitat2kachaka/kachaka-api/python/")


if __name__ == "__main__":
    client = kachaka_api.KachakaApiClient(sys.argv[1])
    print(client.get_locations())
    # 棚をとりにいく
    client.move_shelf("S01", "L01")
    while True:
        move(client, 4.6, 2.5, np.pi)
        move(client, 0.8, 4, np.pi / 2)
