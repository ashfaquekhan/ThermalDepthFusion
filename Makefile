# Makefile for ThermalDepthFusion (thermal base + Arducam ToF depth overlay)

CXX = g++
CXXFLAGS = -std=c++14 -Iinclude -I. -I/usr/include -DLINUX -D_GNU_SOURCE -O2 -march=armv8-a
LDFLAGS  = -Llibs -Wl,-rpath,'$$ORIGIN/libs'

OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv)
OPENCV_LIBS   := $(shell pkg-config --libs   opencv4 2>/dev/null || pkg-config --libs   opencv)

CORE_LIBS    = -lpthread -lusb-1.0 -lm -ldl -lrt -li2c
THERMAL_LIBS = libs/libiruvc.so libs/libirtemp.so libs/libirprocess.so libs/libirparse.so
TOF_LIBS     = -lArducamDepthCamera

TARGET = thermal_depth_fusion
SOURCE = thermal_depth_fusion.cpp

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -o $(TARGET) $(SOURCE) \
		$(LDFLAGS) $(CORE_LIBS) $(THERMAL_LIBS) $(TOF_LIBS) $(OPENCV_LIBS)

# Runs as the pi user (NO sudo): thermal USB access comes from the udev rule
# (99-thermalcam.rules, MODE 0666 on 0bda:5840) and /dev/video2 from the video
# group. Running as pi (not root) gives Qt the correct XDG_RUNTIME_DIR, which
# fixes the OpenCV window freezing on the touchscreen.
run: $(TARGET)
	-pkill -9 -f '$(TARGET)$$'
	-./reset_thermal.sh
	@sleep 1
	DISPLAY=:0 XAUTHORITY=/home/pi/.Xauthority \
	     XDG_RUNTIME_DIR=/run/user/1000 \
	     DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
	     ./$(TARGET) 2>/dev/null | grep --line-buffered -v "UVC: ioctl"

clean:
	rm -f $(TARGET)

.PHONY: all run clean
