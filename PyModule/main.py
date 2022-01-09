import os
from dotenv import load_dotenv
import logging
import json
import traceback
import asyncio

from finger_main import FingerMain

fig_session = None
upper_client = None
fig: FingerMain


def eval_input(input_str, is_json=True):
    try:
        ev = eval(input_str)
        if is_json is False:
            return ev
        else:
            return json.dumps({'status': 0, 'info': 'OK',
                               'error': {}, 'data': ev}, sort_keys=True, indent=4, separators=(',', ': '))

    except BaseException as e:
        return json.dumps({'status': 1, 'info': 'Service Exception', 'error': {
            'name': type(e).__name__,
            'message': str(e),
            'trace': traceback.format_exc()
        }, 'data': {}}, sort_keys=True, indent=4, separators=(',', ': '))


async def main():
    print("=== LabLockerV7 Fingerprint Serive by Kenvix ===")
    load_dotenv()
    logging.basicConfig(level=int(os.getenv("PYEXECUTOR_LOG_LEVEL")),
                        format=os.getenv("PYEXECUTOR_LOG_FORMAT"),
                        datefmt=os.getenv("PYEXECUTOR_LOG_DATE_FORMAT")
                        )
    logging.debug("Bootstrapping")
    global fig_session, upper_client, fig

    if os.getenv("FINGERPRINT_ENABLE") == "1":
        fig = FingerMain(device=os.getenv("FINGERPRINT_DEVICE"), save_path=os.getenv("FINGERPRINT_SAVE_PATH"))
        fig_session = fig.get_session()
        fig.set_security_level(os.getenv("FINGERPRINT_SECURITY_LEVEL"))
        fig.add_fingerprint()
        fig.start_loop_async()

    if os.getenv("PYEXECUTOR_UPPER_ENABLE") == "1":
        import upper_client
        upper_client.run_forever()

    while True:
        print(eval_input(input(), False))

    logging.debug("Normally Shutdown down ...")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    res = loop.run_until_complete(asyncio.wait([main()]))
    loop.close()
