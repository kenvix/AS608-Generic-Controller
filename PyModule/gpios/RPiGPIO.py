import os
import sys

from RPi import GPIO
import asyncio

if os.getenv("LOCKER_GPIO_DRIVER").lower() == "fake-verbose":
    GPIO.VERBOSE = True

IsDoorOperating = False
IsFakeGPIO = os.getenv("LOCKER_GPIO_DRIVER").lower().startswith("fake")
WaitTimeUserDoor = float(os.getenv("LOCKER_GPIO_DOOR_WAIT_USER_DELAY"))
PinBeep = int(os.getenv("LOCKER_GPIO_PIN_BEEP"))


if os.getenv("LOCKER_GPIO_DOOR_DRIVER").lower() == "sg996r":
    import Steering as door
else:
    import DirectDoor as door


async def init():
    GPIO.setmode(GPIO.BOARD)
    await door.door_init()
    GPIO.setup(PinBeep, GPIO.OUT)
    GPIO.output(PinBeep, GPIO.LOW)


async def door_open():
    await door.door_open()


async def door_close():
    await door.door_close()


async def door_unlock():
    global IsDoorOperating
    if IsDoorOperating:
        return

    IsDoorOperating = True
    await door_open()
    await asyncio.sleep(WaitTimeUserDoor)
    await door_close()
    IsDoorOperating = False
    pass


async def beep(voiceDelay=0.3, silentDelay=0.25, times=1):
    for i in range(0, times):
        GPIO.output(PinBeep, GPIO.HIGH)
        await asyncio.sleep(voiceDelay)
        GPIO.output(PinBeep, GPIO.LOW)
        await asyncio.sleep(silentDelay)
        # Also beep user Terminal!
        print('\a', file=sys.stderr)
