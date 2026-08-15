FROM osrf/ros:jazzy-desktop

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ros-jazzy-moveit \
        ros-jazzy-ros-gz \
        ros-jazzy-cv-bridge \
        ros-jazzy-ur \
        ros-jazzy-ur-simulation-gz \
        python3-numpy \
        python3-pil \
    && rm -rf /var/lib/apt/lists/*

ENV ROS_DISTRO=jazzy \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    QT_QPA_PLATFORM=offscreen
