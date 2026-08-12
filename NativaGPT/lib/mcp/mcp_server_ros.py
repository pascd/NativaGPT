#!/usr/bin/env python3
"""
MCP Server for ROS (Unified) - HYBRID BRIDGE MODE
Auto-detects ROS 1 or ROS 2 and bridges to system python for compatibility.

Performance optimizations:
- Cached ROS environment detection
- Connection pooling for subprocess calls
- Pre-warmed shell environment
"""

import os
import sys
import tempfile
import subprocess
import json
import time
import threading
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from NativaGPT.lib.coloring_logger import logger

# === CONFIGURATION & DETECTION ===
SYSTEM_PYTHON = "/usr/bin/python3"

# Cache for ROS environment (thread-safe)
_ros_env_cache = {
    "version": None,
    "distro": None,
    "setup_path": None,
    "detected": False,
}
_ros_env_lock = threading.Lock()


def detect_ros_environment():
    """Detect the active or available ROS distribution, caching the result.

    Checks the ROS_DISTRO/ROS_VERSION environment variables first, then
    falls back to scanning common /opt/ros/<distro>/setup.bash paths (ROS 2
    distros are checked before ROS 1 ones).

    Returns:
        A tuple `(version, distro, setup_path)`, e.g. `(2, 'humble',
        '/opt/ros/humble/setup.bash')`. `setup_path` is None when the
        environment variables were used directly instead of a filesystem
        scan. All three are None if no ROS installation could be found.
    """
    global _ros_env_cache

    with _ros_env_lock:
        if _ros_env_cache["detected"]:
            return (
                _ros_env_cache["version"],
                _ros_env_cache["distro"],
                _ros_env_cache["setup_path"],
            )

    # 1. Check Environment Variables (Active shell) - Fast path
    if "ROS_DISTRO" in os.environ and "ROS_VERSION" in os.environ:
        version = int(os.environ["ROS_VERSION"])
        distro = os.environ["ROS_DISTRO"]
        with _ros_env_lock:
            _ros_env_cache = {
                "version": version,
                "distro": distro,
                "setup_path": None,
                "detected": True,
            }
        return version, distro, None

    # 2. Scan Filesystem (Auto-discovery)
    # Order determines priority (ROS 2 > ROS 1 usually preferred today)
    candidates = [
        (2, "jazzy"),
        (2, "humble"),
        (2, "iron"),
        (2, "foxy"),  # ROS 2
        (1, "noetic"),
        (1, "melodic"),  # ROS 1
    ]

    for version, distro in candidates:
        setup_path = f"/opt/ros/{distro}/setup.bash"
        if os.path.exists(setup_path):
            with _ros_env_lock:
                _ros_env_cache = {
                    "version": version,
                    "distro": distro,
                    "setup_path": setup_path,
                    "detected": True,
                }
            return version, distro, setup_path

    return None, None, None


# Run Detection (cached)
ROS_VERSION, ROS_DISTRO, ROS_SETUP_PATH = detect_ros_environment()

mcp = FastMCP("ros_unified")
TEMP_DIR = Path(tempfile.gettempdir()) / "nativa_mcp_images_unified"
TEMP_DIR.mkdir(exist_ok=True)

# Pre-warm the shell environment for faster subprocess calls
_SHELL_WARMED = False
_SHELL_WARM_LOCK = threading.Lock()


def _warmup_shell():
    """Run a throwaway sourced-shell command once to warm the shell cache; failures are ignored."""
    global _SHELL_WARMED
    try:
        source_cmd = get_source_cmd()
        # Quick test to warm up the shell
        test_cmd = f'bash -c "{source_cmd} && echo warmup"'
        result = subprocess.run(
            test_cmd, capture_output=True, shell=True, timeout=2, text=True
        )
        with _SHELL_WARM_LOCK:
            _SHELL_WARMED = result.returncode == 0 and "warmup" in result.stdout
    except Exception:
        pass


# Start warmup in background
warmup_thread = threading.Thread(target=_warmup_shell, daemon=True)
warmup_thread.start()


def get_source_cmd():
    """Build the bash snippet that sources the detected ROS setup script.

    Returns:
        `"source <setup_path>"` if a setup script was found, otherwise
        `"true"` (a no-op, assuming the environment is already sourced).
    """
    if ROS_SETUP_PATH:
        return f"source {ROS_SETUP_PATH}"
    return "true"  # Assume already sourced if no path found but vars exist


def run_bridge_script(script_content):
    """Write a Python script to a temp file and run it with the system Python inside the sourced ROS environment.

    Args:
        script_content: Full source code of the script to run.

    Returns:
        A tuple `(success, stdout, stderr)` where `success` is True if the
        subprocess exited with code 0. The temp script file is deleted
        afterwards.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_content)
        script_path = f.name

    source_cmd = get_source_cmd()
    full_cmd = f'bash -c "{source_cmd} && {SYSTEM_PYTHON} {script_path}"'

    try:
        # Optimized timeout for faster response
        result = subprocess.run(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            timeout=15,
            text=True,
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    finally:
        try:
            os.unlink(script_path)
        except:
            pass


# Cache for CLI commands to avoid repeated subprocess spawning
_cli_cmd_cache = {}
_CACHE_MAX_SIZE = 100
_CACHE_LOCK = threading.Lock()


def run_cli_command(cmd_str):
    """Run a shell command inside the sourced ROS environment, caching results for a few read-only command patterns.

    Args:
        cmd_str: The shell command to execute (e.g. "rostopic list").

    Returns:
        A tuple `(success, stdout, stderr)`. For a small allow-list of
        read-only topic-listing commands, a cached result younger than 5
        seconds is returned instead of re-running the subprocess.
    """
    # Check cache for simple read-only commands
    cacheable_patterns = [
        "rostopic list",
        "rostopic info",
        "ros2 topic list",
        "ros2 topic info",
    ]
    if any(p in cmd_str for p in cacheable_patterns):
        with _CACHE_LOCK:
            if cmd_str in _cli_cmd_cache:
                cached = _cli_cmd_cache[cmd_str]
                # Return cached if less than 5 seconds old
                if time.time() - cached["timestamp"] < 5:
                    return cached["result"]

    source_cmd = get_source_cmd()
    full_cmd = f'bash -c "{source_cmd} && {cmd_str}"'
    try:
        res = subprocess.run(
            full_cmd, capture_output=True, shell=True, text=True, timeout=5
        )
        result = (res.returncode == 0, res.stdout.strip(), res.stderr.strip())

        # Cache the result
        with _CACHE_LOCK:
            if cmd_str not in _cli_cmd_cache and len(_cli_cmd_cache) < _CACHE_MAX_SIZE:
                _cli_cmd_cache[cmd_str] = {"result": result, "timestamp": time.time()}

        return result
    except Exception as e:
        return (False, "", str(e))


# === TOOLS ===


@mcp.tool()
async def get_system_status() -> str:
    """Check which ROS version is active.

    Returns:
        A status string naming the detected ROS version/distro and the
        system Python used to run bridge scripts, or an error message if no
        ROS installation was found.
    """
    if not ROS_DISTRO:
        return "❌ No ROS installation found on this system."
    return f"✅ Active System: ROS {ROS_VERSION} ({ROS_DISTRO})\nUsing System Python: {SYSTEM_PYTHON}"


@mcp.tool()
async def list_topics() -> str:
    """List all active ROS topics.

    Returns:
        A newline-separated list of topic names (with type annotations on
        ROS 2), or an error message if ROS is unavailable or the underlying
        CLI command failed.
    """
    if not ROS_DISTRO:
        return "ROS not found."

    if ROS_VERSION == 1:
        success, out, err = run_cli_command("rostopic list")
        return out if success else f"Error: {err}"

    elif ROS_VERSION == 2:
        success, out, err = run_cli_command("ros2 topic list -t")
        if success:
            lines = [l.replace("[", "(").replace("]", ")") for l in out.split("\n")]
            return "\n".join(lines)
        return f"Error: {err}"

    return f"Unsupported ROS version: {ROS_VERSION}"


@mcp.tool()
async def get_topic_info(topic_name: str) -> str:
    """Get detailed info about a topic.

    Args:
        topic_name: Name of the ROS topic to inspect.

    Returns:
        The CLI output describing the topic's type, publishers, and
        subscribers, or an error message.
    """
    if not ROS_DISTRO:
        return "ROS not found."

    cmd = (
        f"rostopic info {topic_name}"
        if ROS_VERSION == 1
        else f"ros2 topic info {topic_name} --verbose"
    )
    success, out, err = run_cli_command(cmd)
    return out if success else err


@mcp.tool()
async def read_topic(topic_name: str) -> str:
    """Read one message from a topic (text only).

    Args:
        topic_name: Name of the ROS topic to read from.

    Returns:
        The echoed message text (truncated to 2500 characters), a timeout
        message if no message arrived, or an error message.
    """
    if not ROS_DISTRO:
        return "ROS not found."

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
    """Capture a single frame from a camera topic and save it as a JPEG.

    Generates a short standalone Python bridge script (ROS 1 or ROS 2,
    depending on what was detected) that subscribes to `topic_name`,
    converts the received `sensor_msgs/Image` to OpenCV via `cv_bridge`, and
    writes it to a temp file; the script is then run in a separate
    subprocess via `run_bridge_script`, using the system Python inside the
    sourced ROS environment.

    Args:
        topic_name: Name of the image topic to capture from.

    Returns:
        A JSON string with `status`, `image_path`, and `files` on success,
        or an error message describing a timeout, missing dependency, or
        other failure.
    """
    if not ROS_DISTRO:
        return "ROS not found."

    output_path = TEMP_DIR / f"unified_capture_{int(time.time() * 1000)}.jpg"

    # === SCRIPT GENERATION ===
    script_content = ""

    if ROS_VERSION == 1:
        # NOTE: `script_content` below is NOT a real function definition in
        # this module - it is the literal source of a standalone ROS 1
        # bridge script (built via f-string interpolation) that gets written
        # to a temp file and executed in its own subprocess by
        # run_bridge_script(). The `capture()` function defined inside this
        # string only exists in that child script's namespace, not here.
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
        # NOTE: same as the ROS 1 branch above - `script_content` here is the
        # literal source of a separate, ROS 2-specific bridge script run as
        # its own subprocess via run_bridge_script(). Its `capture()` lives
        # only in that generated script's namespace, so this is not a
        # duplicate definition of the ROS 1 `capture()` above.
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
        return json.dumps(
            {
                "status": "success",
                "image_path": str(output_path),
                "files": [str(output_path)],
                "info": f"Captured via ROS {ROS_VERSION} Bridge",
            }
        )

    if "TIMEOUT" in stdout:
        return f"Error: Timeout waiting for image on '{topic_name}'."
    if "ModuleNotFoundError" in stderr:
        return f"Missing dependencies in SYSTEM python. Run: sudo apt install ros-{ROS_DISTRO}-cv-bridge python3-opencv"

    return f"Capture Failed.\nMode: ROS {ROS_VERSION}\nLog: {stdout}\nError: {stderr}"


def main():
    """CLI entry point: print the detected ROS environment and run the MCP server over stdio."""
    print(f"ROS UNIFIED MCP SERVER", file=sys.stderr)
    if ROS_DISTRO:
        print(f"Detected: ROS {ROS_VERSION} ({ROS_DISTRO}) at {ROS_SETUP_PATH}", file=sys.stderr)
    else:
        print(f"No ROS installation detected in /opt/ros/", file=sys.stderr)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
