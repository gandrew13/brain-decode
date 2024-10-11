import sys
import subprocess
from datetime import datetime



def run_job(hour: str, min: str, cmd: str) -> None:
    '''
    Runs a command line command at a specific time
    '''
    hour = int(hour)
    min = int(min)

    while True:
        now = datetime.now()
        if now.hour == hour and now.minute == min:
            subprocess.run(cmd, shell=True)
            return


def main():
    if len(sys.argv) == 4:
        run_job(*sys.argv[1:])
    else:
        print("Error: invalid number of arguments!")


main()