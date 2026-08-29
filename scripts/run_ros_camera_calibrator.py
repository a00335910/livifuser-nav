"""Run ROS Humble's calibrator without incompatible /usr/local Python wheels.

Ubuntu-TB3 has a pip OpenCV 5 / NumPy 2 stack in /usr/local, while ROS Humble's
cv_bridge is compiled against the Ubuntu OpenCV 4 / NumPy 1 ABI.  Removing only
the /usr/local site-packages entry selects the matching Ubuntu packages without
modifying either installed environment.
"""

from __future__ import annotations

import runpy
import sys

INCOMPATIBLE_SITE_PACKAGES = "/usr/local/lib/python3.10/dist-packages"
CALIBRATOR = "/opt/ros/humble/lib/camera_calibration/cameracalibrator"

sys.path = [path for path in sys.path if path != INCOMPATIBLE_SITE_PACKAGES]
runpy.run_path(CALIBRATOR, run_name="__main__")

