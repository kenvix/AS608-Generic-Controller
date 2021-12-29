import os
import logging
import _thread

import websocket


def on_message(ws, message):
    logging.debug(ws)
    logging.debug(message)


def on_error(ws, error):
    logging.debug(ws)
    logging.debug(error)


def on_close(ws):
    logging.debug(ws)
    logging.debug("### closed ###")


def run_forever():
    websocket.enableTrace(True)
    while True:
        try:
            ws = websocket.WebSocketApp(os.getenv("PYEXECUTOR_UPPER_URL"),
                                        on_message=on_message,
                                        on_error=on_error,
                                        on_close=on_close)

            ws.run_forever()
        except Exception as e:
            logging.error(e, exc_info=True)


def run_forever_async():
    _thread.start_new_thread(run_forever, ("Upper-Websocket", 2,))
