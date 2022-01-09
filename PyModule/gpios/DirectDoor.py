import os
from RPi import GPIO
import asyncio

if os.getenv("LOCKER_GPIO_DRIVER").lower() == "fake-verbose":
    GPIO.VERBOSE = True


async def door_init():
    pass


async def door_open():
    pass


async def door_close():
    pass
