import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if "win32" in sys.platform or "cygwin" in sys.platform:
    IS_WINDOWS = True
    COPY_CMD = "robocopy"

else:
    IS_WINDOWS = False
    COPY_CMD = "cp"
