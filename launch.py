import os
import subprocess
import sys

from dotenv import load_dotenv

# Read in .env file; error if no`t found
env_file = ".env"
if not os.path.isfile(env_file):
    print(".env not found, running setup.py")
    subprocess.run([sys.executable, "setup.py"])

# Source .env file
load_dotenv(".env")


def showhelp():
    # Display Help
    print()
    print("Usage: python script.py [option...]")
    print("options:")
    print("  -h       Print this help.")
    print("  -d       Start in daemon mode.")
    print()


options = ""

# Process command-line options
args = sys.argv[1:]
while args:
    option = args.pop(0)
    if option == "-h":
        showhelp()
        sys.exit()
    elif option == "-d":
        options = "-d"
    else:
        print("Error: Invalid option")
        sys.exit()

# On versions above 20.10.2, docker-compose is docker compose
docker_version = subprocess.run(["docker", "--version"], capture_output=True, text=True).stdout


def fix_version_number(version_number):
    # Fix issue with format with extra comma, eg. 24.0.2,
    if version_number[-1] == ",":
        return version_number[:-1]
    return version_number


version_numbers = [float(fix_version_number(v)) for v in docker_version.split()[2].split(".")[:3]]

smaller = min(version_numbers + [20.10, 2])

if smaller == 20.10 or smaller == 2:
    command = "docker " + options + " --remove-orphans --build"
else:
    command = "docker-compose up "+options+" --remove-orphans --build"
subprocess.run(command, shell=True)
