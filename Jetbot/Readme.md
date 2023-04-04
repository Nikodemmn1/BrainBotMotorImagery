


## How to run jetbot code:
 * ssh jetson@ip_addres -p pass
 * python3 Documents/JetbotProject/control/robot_control.py

## If Modified copy files to the robot
 * scp Jetbot/control/robot_control.py jetson@192.168.0.240:/home/jetson/Documents/JetbotProject/control/robot_control.py
 * scp Jetbot/control/robot_communication.py jetson@192.168.0.240:/home/jetson/Documents/JetbotProject/control/robot_communication.py


## FIX na błędy z GSTREAMER:
rm -rf ~/.cache/gstreamer-1.0
source ~/.bashrc
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1




