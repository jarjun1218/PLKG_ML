import threading
import serial
import glob
import time
from datetime import datetime

port = '/dev/ttyUSB1'
baudrate = 115200

Data_Path = r"./collect_data"
Defult = r"csi_data"
filetype = r".txt"

stop_event = threading.Event()

def connect_serial(port, baudrate): 
    timeout = 1  # set timeout

    # open serial
    ser = serial.Serial(port, baudrate, timeout = timeout)

    return ser

def generate_filename():
    global Data_Path

    path = Data_Path + r"/" + Defult + r"*"+ filetype
    filelist = glob.glob(path)
    if len(filelist) == 0:
        index = 1
    else:
        index = 1
        for s in filelist:
            temp = int(s[len(Data_Path)+len(Defult)+1:-len(filetype)]) + 1
            index = max(temp,index)
    output_file =  Data_Path + r"/" + Defult + str(index) + filetype

    return output_file

def read_csi(ser):
    file = generate_filename()

    with open(file, 'a') as f:
        print(f"Start Collecting...")
        print(f"Writing CSI data in", file)
        try:
            while not stop_event.is_set():
                line = ser.readline().decode().strip()
                if "serial_num:" in line:
                    line = str(datetime.now()) + " "+ line
                    f.write(line + '\n\n')
                    #print(f"寫入: {line}")
        except KeyboardInterrupt:
            print(f"Stop Collecting...")
            f.close()
    f.close()

def main():
    ser = connect_serial(port, baudrate)
    task = None

    while True:
        t = input("\n>>")
        if t == "r" and (task is None or not task.is_alive()):
            stop_event.clear()
            task = threading.Thread(target=read_csi, args=(ser,))
            task.start()
        elif t == "q" and task is not None:
            stop_event.set()
            task.join()
            print(f"Stop Collecting...")
            #print(f"Serial closed...")
        elif t == 'e':
            ser.close()
            break
        else:
            pass
        time.sleep(1)

if __name__ == "__main__":
    main()
