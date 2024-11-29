import sys
import subprocess
from datetime import datetime



def run_job(day: str, hour: str, min: str, cmd: str) -> None:
    '''
    Runs a command line command at a specific time.
    '''
    day = int(day)
    hour = int(hour)
    min = int(min)

    while True:
        now = datetime.now()
        if now.day == day and now.hour == hour and now.minute == min:
            subprocess.run(cmd, shell=True)
            return


def main():
    if len(sys.argv) == 5:      # [runner.py day hour minute cmd]
        run_job(*sys.argv[1:])
    else:
        print("Error: invalid number of arguments!")


main()