import os
from RPi import GPIO
import asyncio

if os.getenv("LOCKER_GPIO_DRIVER").lower() == "fake-verbose":
    GPIO.VERBOSE = True

PinOpen = int(os.getenv("LOCKER_GPIO_PIN_OPEN"))
PinClose = int(os.getenv("LOCKER_GPIO_PIN_CLOSE"))
WaitTime = float(os.getenv("LOCKER_GPIO_DOOR_ROLLTATE_DELAY"))


async def door_init():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(PinOpen, GPIO.OUT)
    GPIO.output(PinOpen, GPIO.LOW)
    GPIO.setup(PinClose, GPIO.OUT)
    GPIO.output(PinClose, GPIO.LOW)


async def door_open():
    GPIO.output(PinOpen, GPIO.HIGH)
    await asyncio.sleep(WaitTime)
    GPIO.output(PinOpen, GPIO.LOW)


async def door_close():
    GPIO.output(PinClose, GPIO.HIGH)
    await asyncio.sleep(WaitTime)
    GPIO.output(PinClose, GPIO.LOW)
