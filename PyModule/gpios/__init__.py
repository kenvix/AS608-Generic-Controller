import os
from RPi import GPIO
import asyncio

if os.getenv("LOCKER_GPIO_DRIVER").lower() == "fake-verbose":
    GPIO.VERBOSE = True

if os.getenv("LOCKER_GPIO_DOOR_DRIVER").lower() == "sg996r":
    import SG996R as door
else:
    import DirectDoor as door
