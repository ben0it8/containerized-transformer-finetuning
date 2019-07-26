from subprocess import call
import os

RESOURCES_DIR = os.getenv("RESOURCES_PATH", "/resources")

call("python " + RESOURCES_DIR + "/app.py", shell=True)
    