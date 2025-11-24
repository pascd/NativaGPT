#!/usr/bin/env python3
"""
MCP Server for ROS1 - BULLETPROOF AUTO-DETECTION (2025 Edition)
Works inside venv, conda, poetry, docker — no sourcing needed!
"""

import json
import os
import sys
import tempfile
import subprocess
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# === AUTO-FIND ROS EVEN IF NOT SOURCED ===
def find_ros_distro():
    """Find any installed ROS1 distro (noetic, melodic, kinetic...)"""
    candidates = ["/opt/ros/noetic", "/opt/ros/melodic", "/opt/ros/lunar", "/opt/ros/kinetic"]
    for path in candidates:
        if Path(path).exists():
            return path.replace("/opt/ros/", "")
    return None

def auto_source_ros():
    """Inject ROS environment into current process - works in venv!"""
    distro = find_ros_distro()
    if not distro:
        print("ROS1 not found in /opt/ros/", file=sys.stderr)
        return False

    setup_path = f"/opt/ros/{distro}/setup.bash"
    if not Path(setup_path).exists():
        print(f"setup.bash not found: {setup_path}", file=sys.stderr)
        return False

    # This is the magic: source the bash file and capture environment
    cmd = f'bash -c "source {setup_path} && env"'
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
    for line in proc.stdout:
        line = line.decode().strip()
        if '=' in line and not line.startswith('_'):
            key, value = line.split('=', 1)
            os.environ[key] = value

    # Also source workspace if it exists
    ws_paths = [
        "/home/$USER/catkin_ws/devel/setup.bash",
        "/home/$USER/ros_ws/devel/setup.bash",
        "./devel/setup.bash",
        "../devel/setup.bash",
    ]
    for ws in ws_paths:
        ws_path = os.path.expanduser(os.path.expandvars(ws))
        if Path(ws_path).exists():
            cmd = f'bash -c "source {ws_path} && env"'
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
            for line in proc.stdout:
                line = line.decode().strip()
                if '=' in line and not line.startswith('_'):
                    key, value = line.split('=', 1)
                    os.environ[key] = value
            print(f"Auto-sourced workspace: {ws_path}", file=sys.stderr)
            break

    print(f"ROS {distro} auto-sourced successfully!", file=sys.stderr)
    return True

# === RUN AUTO-SOURCE BEFORE ANY ROS IMPORT ===
ROS_READY = auto_source_ros()

# Now safely import ROS
try:
    import rospy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    import rostopic
    HAS_ROSPY = True
    bridge = CvBridge()
except Exception as e:
    rospy = None
    HAS_ROSPY = False
    print(f"Failed to import ROS packages: {e}", file=sys.stderr)

HAS_CV2 = False
try:
    import cv2
    HAS_CV2 = True
except:
    pass

mcp = FastMCP("ros1")
TEMP_DIR = Path(tempfile.gettempdir()) / "ros_mcp_images"
TEMP_DIR.mkdir(exist_ok=True)

# === YOUR TOOLS (unchanged) ===
@mcp.tool()
async def list_topics() -> str:
    if not HAS_ROSPY:
        return "ROS not available"
    topics = rospy.get_published_topics()
    return "\n".join([f"{t[0]} ({t[1]})" for t in sorted(topics)])

@mcp.tool()
async def capture_camera_image(topic_name: str = "/camera_rgb/image_raw") -> str:
    if not HAS_ROSPY or not HAS_CV2:
        return "ROS or OpenCV not available"
    try:
        msg = rospy.wait_for_message(topic_name, Image, timeout=5)
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        path = TEMP_DIR / f"capture_{int(rospy.Time.now().to_sec()*1000)}.jpg"
        cv2.imwrite(str(path), cv_img)
        return f"Image saved: {path}\nSize: {path.stat().st_size//1024}KB"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def read_topic(topic_name: str, timeout: float = 3.0) -> str:
    if not HAS_ROSPY:
        return "ROS not available"
    try:
        msg_class, real_topic, _ = rostopic.get_topic_class(topic_name)
        if msg_class is None:
            return f"Topic {topic_name} not found"
        msg = rospy.wait_for_message(real_topic, msg_class, timeout=timeout)
        return str(msg)
    except Exception as e:
        return f"Error reading {topic_name}: {str(e)}"

@mcp.tool()
async def get_topic_info(topic_name: str) -> str:
    if not HAS_ROSPY:
        return "ROS not available"
    try:
        topic_type, _, _ = rostopic.get_topic_class(topic_name, blocking=False)
        state = rospy.get_master().getSystemState()
        pubs = [n for t, n in state[0] if t == topic_name]
        subs = [n for t, n in state[1] if t == topic_name]
        return f"Topic: {topic_name}\nType: {topic_type._type if topic_type else 'unknown'}\nPublishers: {pubs}\nSubscribers: {subs}"
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print("="*60, file=sys.stderr)
    print("ROS1 MCP Server - BULLETPROOF MODE", file=sys.stderr)
    print("="*60, file=sys.stderr)
    print(f"ROS ready: {HAS_ROSPY}", file=sys.stderr)
    if HAS_ROSPY:
        print(f"Node will run as: {rospy.get_name()}", file=sys.stderr)
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()