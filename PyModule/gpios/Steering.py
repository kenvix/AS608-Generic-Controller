import os
from RPi import GPIO
import asyncio

if os.getenv("LOCKER_GPIO_DRIVER").lower() == "fake-verbose":
    GPIO.VERBOSE = True


PinControl = int(os.getenv("LOCKER_GPIO_PIN_CONTROL"))
WaitTimeA = float(os.getenv("LOCKER_GPIO_DOOR_CONTROL_OPEN_DELAY"))
WaitTimeB = float(os.getenv("LOCKER_GPIO_DOOR_CONTROL_CLOSE_DELAY"))


async def door_init():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(PinControl, GPIO.OUT)
    GPIO.output(PinControl, GPIO.LOW)


async def door_open():
    GPIO.output(PinControl, GPIO.HIGH)
    await asyncio.sleep(WaitTimeA)
    GPIO.output(PinControl, GPIO.LOW)


async def door_close():
    GPIO.output(PinControl, GPIO.HIGH)
    await asyncio.sleep(WaitTimeB)
    GPIO.output(PinControl, GPIO.LOW)
