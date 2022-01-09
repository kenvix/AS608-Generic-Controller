import os
from RPi import GPIO
import asyncio

if os.getenv("LOCKER_GPIO_DRIVER").lower() == "fake-verbose":
    GPIO.VERBOSE = True

IsDoorOperating = False

if os.getenv("LOCKER_GPIO_DOOR_DRIVER").lower() == "sg996r":
    import SG996R as door
else:
    import DirectDoor as door


async def door_init():
    await door.door_init()


async def door_open():
    await door.door_open()


async def door_close():
    await door.door_close()


async def door_unlock():
    await door_open()
    await asyncio.sleep(int(os.getenv("LOCKER_GPIO_DOOR_WAIT_USER_DELAY")))
    await door_close()
    pass
