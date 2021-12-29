import os

import as608_combo_lib as as608


def connect():
    session = as608.connect_serial_session(os.getenv("FINGERPRINT_DEVICE"))
    return session
