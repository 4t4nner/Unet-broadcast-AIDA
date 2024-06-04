import sys
import cv2
import socket
import numpy
import time
from datetime import datetime, timezone
import base64
import subprocess
from os import path, chdir

def check_download(video_path):
    if not (path.isfile(video_path) or path.isfile('model_scripted.pt')):
        subprocess.run(["pip","install", "gdown"])
        subprocess.run(["gdown", "1gfJ1cBU_R7JbscQfmheoGBMzkFag0YMU"])

dir = path.dirname(path.abspath(__file__))
chdir(dir)
video_path = path.join(dir, 'out.mp4')
check_download(video_path)


class ClientVideoSocket:
    def __init__(self, ip, port, video_path):
        self.TCP_SERVER_IP = ip
        self.TCP_SERVER_PORT = port
        self.video_path = video_path
        self.connectCount = 0
        self.connectServer()

    def connectServer(self):
        try:
            self.sock = socket.socket()
            self.sock.connect((self.TCP_SERVER_IP, self.TCP_SERVER_PORT))
            print(u'Client socket is connected with Server socket [ TCP_SERVER_IP: ' + self.TCP_SERVER_IP + ', TCP_SERVER_PORT: ' + str(self.TCP_SERVER_PORT) + ' ]')
            self.connectCount = 0
            self.sendImages()
        except Exception as e:
            print(e)
            self.connectCount += 1
            if self.connectCount == 10:
                print(u'Connect fail %d times. exit program'%(self.connectCount))
                sys.exit()
            print(u'%d times try to connect with server'%(self.connectCount))
            time.sleep(1)
            self.connectServer()

    def sendImages(self):
        cnt = 0
        capture = cv2.VideoCapture(self.video_path)

        #  здесь можно воткнуть получение инфы о файле из ffmpeg
        #  примерно так: ffmpeg -i out.mp4 -f ffmetadata
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        try:
            while capture.isOpened():
                ret, frame = capture.read()
                # resize_frame = cv2.resize(frame, dsize=(480, 315), interpolation=cv2.INTER_AREA)

                now = time.localtime()
                stime = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')

                encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
                result, imgencode = cv2.imencode('.jpg', frame, encode_param)
                data = numpy.array(imgencode)
                stringData = base64.b64encode(data)
                length = str(len(stringData))
                self.sock.sendall(length.encode('utf-8').ljust(64))
                self.sock.send(stringData)
                self.sock.send(stime.encode('utf-8').ljust(64))
                print(u'send images %d'%(cnt))
                cnt+=1
                time.sleep(0.02)
        except Exception as e:
            print(e)
            self.sock.close()
            time.sleep(1)
            self.connectServer()
            self.sendImages()

def main():
    TCP_IP = 'localhost'
    TCP_PORT = 8081

    dir = path.dirname(path.abspath(__file__))
    chdir(dir)
    video_path = path.join(dir, 'out.mp4')

    client = ClientVideoSocket(TCP_IP, TCP_PORT, video_path)

if __name__ == "__main__":
    main()