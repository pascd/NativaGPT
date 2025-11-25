#!/usr/bin/env python3
"""
MCP Server for ROS (Unified) - HYBRID BRIDGE MODE
Auto-detects ROS 1 or ROS 2 and bridges to system python for compatibility.
"""

import os
import sys
import tempfile
import subprocess
import json
import time
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from NativaGPT.lib.coloring_logger import logger

# === CONFIGURATION & DETECTION ===
SYSTEM_PYTHON = "/usr/bin/python3"

def detect_ros_environment():
    """
    Detects active or available ROS distribution.
    Returns: (version_int, distro_name, setup_path)
    Example: (2, 'humble', '/opt/ros/humble/setup.bash')
    """
    # 1. Check Environment Variables (Active shell)
    if 'ROS_DISTRO' in os.environ and 'ROS_VERSION' in os.environ:
        return int(os.environ['ROS_VERSION']), os.environ['ROS_DISTRO'], None

    # 2. Scan Filesystem (Auto-discovery)
    # Order determines priority (ROS 2 > ROS 1 usually preferred today)
    candidates = [
        (2, "jazzy"), (2, "humble"), (2, "iron"), (2, "foxy"),  # ROS 2
        (1, "noetic"), (1, "melodic")                           # ROS 1
    ]

    for version, distro in candidates:
        setup_path = f"/opt/ros/{distro}/setup.bash"
        if os.path.exists(setup_path):
            return version, distro, setup_path

    return None, None, None

# Run Detection
ROS_VERSION, ROS_DISTRO, ROS_SETUP_PATH = detect_ros_environment()

mcp = FastMCP("ros_unified")
TEMP_DIR = Path(tempfile.gettempdir()) / "nativa_mcp_images_unified"
TEMP_DIR.mkdir(exist_ok=True)

def get_source_cmd():
    """Returns the bash command to source the correct environment."""
    if ROS_SETUP_PATH:
        return f"source {ROS_SETUP_PATH}"
    return "true" # Assume already sourced if no path found but vars exist

def run_bridge_script(script_content):
    """Writes and runs a python script using the System Python + ROS Env."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name

    source_cmd = get_source_cmd()
    full_cmd = f'bash -c "{source_cmd} && {SYSTEM_PYTHON} {script_path}"'

    try:
        # Increased timeout for ROS 2 discovery
        result = subprocess.run(
            full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            shell=True, timeout=15, text=True
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    finally:
        try:
            os.unlink(script_path)
        except:
            pass

def run_cli_command(cmd_str):
    """Runs a shell command inside the ROS environment."""
    source_cmd = get_source_cmd()
    full_cmd = f'bash -c "{source_cmd} && {cmd_str}"'
    try:
        res = subprocess.run(full_cmd, capture_output=True, shell=True, text=True, timeout=5)
        return res.returncode == 0, res.stdout.strip(), res.stderr.strip()
    except Exception as e:
        return False, "", str(e)

# === TOOLS ===

@mcp.tool()
async def get_system_status() -> str:
    """Check which ROS version is active."""
    if not ROS_DISTRO:
        return "❌ No ROS installation found on this system."
    return f"✅ Active System: ROS {ROS_VERSION} ({ROS_DISTRO})\nUsing System Python: {SYSTEM_PYTHON}"

@mcp.tool()
async def list_topics() -> str:
    """List all active ROS topics."""
    if not ROS_DISTRO: return "ROS not found."

    if ROS_VERSION == 1:
        success, out, err = run_cli_command("rostopic list")
        # Enhance with types if possible (slower but useful)
        return out if success else f"Error: {err}"

    elif ROS_VERSION == 2:
        # ROS 2 supports types directly in list
        success, out, err = run_cli_command("ros2 topic list -t")
        if success:
            # Clean up formatting: "topic [type]" -> "topic (type)"
            lines = [l.replace('[', '(').replace(']', ')') for l in out.split('\n')]
            return "\n".join(lines)
        return f"Error: {err}"

@mcp.tool()
async def get_topic_info(topic_name: str) -> str:
    """Get detailed info about a topic."""
    if not ROS_DISTRO: return "ROS not found."

    cmd = f"rostopic info {topic_name}" if ROS_VERSION == 1 else f"ros2 topic info {topic_name} --verbose"
    success, out, err = run_cli_command(cmd)
    return out if success else err

@mcp.tool()
async def read_topic(topic_name: str) -> str:
    """Read one message from a topic (text only)."""
    if not ROS_DISTRO: return "ROS not found."

    cmd = ""
    if ROS_VERSION == 1:
        cmd = f"rostopic echo -n 1 {topic_name}"
    else:
        cmd = f"ros2 topic echo --once {topic_name}"

    success, out, err = run_cli_command(cmd)
    if not success and "timeout" not in err.lower():
        return f"Error reading topic: {err}"
    if not out:
        return "Timeout: No message received."

    return out[:2500] + ("..." if len(out) > 2500 else "")

@mcp.tool()
async def capture_camera_image(topic_name: str = "/camera/color/image_raw") -> str:
    """
    Captures an image using a Unified System Bridge.
    Works for both ROS 1 and ROS 2 by injecting the correct script.
    """
    if not ROS_DISTRO: return "ROS not found."

    output_path = TEMP_DIR / f"unified_capture_{int(time.time()*1000)}.jpg"

    # === SCRIPT GENERATION ===
    script_content = ""

    if ROS_VERSION == 1:
        script_content = f"""
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import sys

def capture():
    try:
        rospy.init_node('nativa_unified_snap', anonymous=True)
        print("WAITING")
        msg = rospy.wait_for_message('{topic_name}', Image, timeout=5.0)

        bridge = CvBridge()
        # Handle encoding logic
        if hasattr(msg, 'encoding') and '8UC1' in msg.encoding:
             cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        else:
             cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        cv2.imwrite('{str(output_path)}', cv_img)
        print("SUCCESS")
    except Exception as e:
        print(f"ERROR: {{e}}")

if __name__ == "__main__":
    capture()
"""
    elif ROS_VERSION == 2:
        script_content = f"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import sys

def capture():
    rclpy.init()
    node = rclpy.create_node('nativa_unified_snap')

    future = rclpy.task.Future()
    def cb(msg):
        if not future.done(): future.set_result(msg)

    # Best effort QoS for cameras
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

    try:
        node.create_subscription(Image, '{topic_name}', cb, qos)
        print("WAITING")
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)

        if future.done() and future.result():
            msg = future.result()
            bridge = CvBridge()
            if '8UC1' in msg.encoding:
                 cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            else:
                 cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv2.imwrite('{str(output_path)}', cv_img)
            print("SUCCESS")
        else:
            print("TIMEOUT")
    except Exception as e:
        print(f"ERROR: {{e}}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    capture()
"""

    # === EXECUTION ===
    success, stdout, stderr = run_bridge_script(script_content)

    if "SUCCESS" in stdout and output_path.exists():
        size_kb = output_path.stat().st_size // 1024
        return json.dumps({
            "status": "success",
            "image_path": str(output_path),
            "files": [str(output_path)],
            "info": f"Captured via ROS {ROS_VERSION} Bridge"
        })

    if "TIMEOUT" in stdout:
        return f"Error: Timeout waiting for image on '{topic_name}'."
    if "ModuleNotFoundError" in stderr:
        return f"Missing dependencies in SYSTEM python. Run: sudo apt install ros-{ROS_DISTRO}-cv-bridge python3-opencv"

    return f"Capture Failed.\nMode: ROS {ROS_VERSION}\nLog: {stdout}\nError: {stderr}"

def main():
    logger.info(f"ROS UNIFIED MCP SERVER")
    if ROS_DISTRO:
        logger.info(f"Detected: ROS {ROS_VERSION} ({ROS_DISTRO}) at {ROS_SETUP_PATH}")
    else:
        logger.info(f"No ROS installation detected in /opt/ros/")
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()