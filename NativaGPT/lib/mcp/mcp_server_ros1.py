#!/usr/bin/env python3
"""
MCP Server for ROS1 Topics - Complete Edition
Handles all data types including images, point clouds, and large arrays
"""

import json
import subprocess
import base64
import tempfile
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml

from mcp.server.fastmcp import FastMCP

# Try to import optional dependencies
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from sensor_msgs.msg import Image as RosImage
    from cv_bridge import CvBridge
    HAS_CV_BRIDGE = False  # We'll use subprocess instead
except ImportError:
    HAS_CV_BRIDGE = False

# Initialize FastMCP server
mcp = FastMCP("ros1_topics_complete")

# Temporary directory for extracted data
TEMP_DIR = Path(tempfile.gettempdir()) / "ros_mcp_data"
TEMP_DIR.mkdir(exist_ok=True)


def run_ros_command(command: List[str], timeout: float = 10.0) -> tuple:
    """Run a ROS command and return success status and output."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout} seconds"
    except Exception as e:
        return False, f"Error running command: {str(e)}"


def check_ros_environment() -> tuple:
    """Check if ROS is properly sourced."""
    success, output = run_ros_command(['rostopic', 'list'], timeout=5.0)
    if not success:
        return False, "ROS environment not found"
    return True, "ROS environment OK"


@mcp.tool()
async def list_topics() -> str:
    """List all available ROS1 topics with their message types."""
    env_ok, env_msg = check_ros_environment()
    if not env_ok:
        return env_msg

    success, output = run_ros_command(['rostopic', 'list'])
    if not success:
        return f"Error listing topics: {output}"

    topics = [t.strip() for t in output.strip().split('\n') if t.strip()]
    if not topics:
        return "No topics found."

    result = ["Available ROS1 Topics:", "=" * 60]

    for topic in sorted(topics):
        success, type_output = run_ros_command(['rostopic', 'type', topic], timeout=2.0)
        topic_type = type_output.strip() if success else "Unknown"
        result.append(f"\nTopic: {topic}")
        result.append(f"  Type: {topic_type}")

    result.append(f"\nTotal topics: {len(topics)}")
    return "\n".join(result)


@mcp.tool()
async def get_topic_info(topic_name: str) -> str:
    """Get detailed information about a specific ROS1 topic."""
    success, output = run_ros_command(['rostopic', 'info', topic_name])
    if not success:
        success, list_output = run_ros_command(['rostopic', 'list'])
        if success:
            return f"Topic '{topic_name}' not found.\n\nAvailable topics:\n{list_output}"
        return f"Error: {output}"
    return f"Topic Information: {topic_name}\n{'=' * 60}\n{output}"


@mcp.tool()
async def capture_image_from_topic(topic_name: str, save_path: Optional[str] = None) -> str:
    """Capture an image from a ROS image topic and save it.

    Args:
        topic_name: Full name of the image topic (e.g., "/camera/color/image_raw")
        save_path: Optional path to save image (default: auto-generated in temp dir)

    Returns path to saved image and metadata.
    """
    # Get one message
    success, output = run_ros_command(
        ['rostopic', 'echo', '-n', '1', topic_name],
        timeout=10.0
    )

    if not success:
        return f"Could not capture image: {output}"

    # Parse YAML output
    try:
        data = yaml.safe_load(output)
    except Exception as e:
        return f"Could not parse image data: {e}"

    if not isinstance(data, dict):
        return "Invalid image message format"

    # Extract image metadata
    height = data.get('height', 0)
    width = data.get('width', 0)
    encoding = data.get('encoding', 'unknown')
    step = data.get('step', 0)
    image_data = data.get('data', [])

    if not image_data:
        return "No image data found in message"

    # Generate save path
    if not save_path:
        timestamp = subprocess.run(['date', '+%Y%m%d_%H%M%S'],
                                  capture_output=True, text=True).stdout.strip()
        save_path = str(TEMP_DIR / f"image_{timestamp}.png")

    # Convert data to image
    if HAS_CV2:
        try:
            # Convert to numpy array
            img_array = np.array(image_data, dtype=np.uint8)

            # Reshape based on encoding
            if encoding == 'rgb8':
                img = img_array.reshape((height, width, 3))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif encoding == 'bgr8':
                img = img_array.reshape((height, width, 3))
            elif encoding == 'mono8':
                img = img_array.reshape((height, width))
            else:
                return f"Unsupported encoding: {encoding}"

            # Save image
            cv2.imwrite(save_path, img)

            result = {
                'success': True,
                'image_path': save_path,
                'width': width,
                'height': height,
                'encoding': encoding,
                'size_bytes': len(image_data),
                'message': f"Image saved to {save_path}"
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return f"Error processing image: {e}"
    else:
        # Fallback: save raw data and metadata
        meta_path = save_path.replace('.png', '_meta.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'width': width,
                'height': height,
                'encoding': encoding,
                'data_length': len(image_data)
            }, f, indent=2)

        return f"OpenCV not available. Metadata saved to {meta_path}. Install opencv-python for image processing."


@mcp.tool()
async def analyze_image_topic(topic_name: str) -> str:
    """Analyze an image from a ROS topic and return visual description.

    Args:
        topic_name: Full name of the image topic

    Returns analysis of the image content.
    """
    # First capture the image
    capture_result = await capture_image_from_topic(topic_name)

    try:
        result_data = json.loads(capture_result)
        if not result_data.get('success'):
            return capture_result

        image_path = result_data['image_path']
    except:
        return capture_result

    if not HAS_CV2:
        return f"Image captured at {image_path}. Install opencv-python for analysis."

    # Analyze image
    try:
        img = cv2.imread(image_path)
        if img is None:
            return f"Could not load image from {image_path}"

        height, width = img.shape[:2]
        channels = img.shape[2] if len(img.shape) > 2 else 1

        # Calculate statistics
        mean_color = np.mean(img, axis=(0, 1)) if channels > 1 else np.mean(img)
        std_color = np.std(img, axis=(0, 1)) if channels > 1 else np.std(img)

        # Detect brightness
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if channels > 1 else img
        brightness = np.mean(gray)

        # Detect edges (for complexity)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)

        # Color distribution
        if channels > 1:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hue_mean = np.mean(hsv[:,:,0])
            saturation_mean = np.mean(hsv[:,:,1])
        else:
            hue_mean = saturation_mean = 0

        analysis = {
            'image_path': image_path,
            'dimensions': f"{width}x{height}",
            'channels': channels,
            'brightness': f"{brightness:.1f}/255 ({'bright' if brightness > 128 else 'dark'})",
            'edge_density': f"{edge_density*100:.1f}% ({'complex' if edge_density > 0.1 else 'simple'} scene)",
            'mean_color_bgr': [float(c) for c in mean_color] if channels > 1 else float(mean_color),
            'color_variation': f"{float(np.mean(std_color)):.1f} ({'high' if np.mean(std_color) > 50 else 'low'} variation)",
            'saturation': f"{saturation_mean:.1f}/255 ({'colorful' if saturation_mean > 100 else 'muted'})" if channels > 1 else "N/A"
        }

        return json.dumps(analysis, indent=2)

    except Exception as e:
        return f"Error analyzing image: {e}"


@mcp.tool()
async def get_image_topic_metadata(topic_name: str) -> str:
    """Get metadata about an image topic without downloading the full image.

    Args:
        topic_name: Full name of the image topic

    Returns image dimensions, encoding, and rate.
    """
    success, output = run_ros_command(
        ['rostopic', 'echo', '-n', '1', topic_name],
        timeout=5.0
    )

    if not success:
        return f"Could not read image topic: {output}"

    result = [f"Image Topic: {topic_name}", "=" * 60]

    # Extract metadata from YAML
    for line in output.split('\n'):
        if any(k in line for k in ['height:', 'width:', 'encoding:', 'step:', 'is_bigendian:']):
            result.append(line.strip())

    # Get frequency
    hz_success, hz_output = run_ros_command(['timeout', '2', 'rostopic', 'hz', topic_name], timeout=3.0)
    if hz_success:
        result.append("\nPublishing Rate:")
        result.append(hz_output)

    return "\n".join(result)


@mcp.tool()
async def sample_topic_data(topic_name: str, samples: int = 1, max_output_size: int = 2000) -> str:
    """Get sample messages from a topic with smart truncation for large data.

    Args:
        topic_name: Full name of the topic
        samples: Number of samples (default: 1, max: 5)
        max_output_size: Max characters to return (default: 2000)

    Returns sample messages, truncated if necessary.
    """
    samples = min(samples, 5)

    success, output = run_ros_command(
        ['rostopic', 'echo', '-n', str(samples), topic_name],
        timeout=10.0
    )

    if not success:
        return f"Could not sample topic: {output}"

    # Check if it's image data (too large)
    if 'encoding:' in output and 'data: [' in output:
        return f"This appears to be an image topic. Use 'analyze_image_topic' or 'get_image_topic_metadata' instead."

    # Truncate if too long
    if len(output) > max_output_size:
        truncated = output[:max_output_size]
        return f"Sample from {topic_name} (truncated from {len(output)} to {max_output_size} chars):\n{'=' * 60}\n{truncated}\n\n... (truncated)"

    return f"Sample from {topic_name}:\n{'=' * 60}\n{output}"


@mcp.tool()
async def analyze_pointcloud_topic(topic_name: str) -> str:
    """Analyze a point cloud topic and return statistics (without full data).

    Args:
        topic_name: Full name of the point cloud topic

    Returns statistics about the point cloud.
    """
    success, output = run_ros_command(
        ['rostopic', 'echo', '-n', '1', topic_name],
        timeout=10.0
    )

    if not success:
        return f"Could not read point cloud: {output}"

    try:
        data = yaml.safe_load(output)
    except Exception as e:
        return f"Could not parse point cloud data: {e}"

    if not isinstance(data, dict):
        return "Invalid point cloud format"

    # Extract metadata
    height = data.get('height', 0)
    width = data.get('width', 0)
    point_step = data.get('point_step', 0)
    row_step = data.get('row_step', 0)
    fields = data.get('fields', [])
    is_dense = data.get('is_dense', False)

    num_points = height * width
    data_size = len(data.get('data', []))

    field_names = [f.get('name', 'unknown') for f in fields] if fields else []

    result = {
        'topic': topic_name,
        'num_points': num_points,
        'dimensions': f"{width}x{height}",
        'point_step': point_step,
        'row_step': row_step,
        'fields': field_names,
        'is_dense': is_dense,
        'data_size_bytes': data_size,
        'estimated_size_mb': f"{data_size / 1024 / 1024:.2f}",
    }

    return json.dumps(result, indent=2)


@mcp.tool()
async def get_topic_summary(topic_name: str) -> str:
    """Get a comprehensive summary of any topic without dumping raw data.

    Args:
        topic_name: Full name of the topic

    Returns summary with type, rate, structure, and intelligent content analysis.
    """
    # Get topic type
    type_success, type_output = run_ros_command(['rostopic', 'type', topic_name])
    topic_type = type_output.strip() if type_success else "Unknown"

    # Check if it's an image topic
    if 'Image' in topic_type:
        return await get_image_topic_metadata(topic_name)

    # Check if it's a point cloud
    if 'PointCloud' in topic_type:
        return await analyze_pointcloud_topic(topic_name)

    # For other topics, do standard summary
    info_success, info_output = run_ros_command(['rostopic', 'info', topic_name])
    def_success, def_output = run_ros_command(['rosmsg', 'show', topic_type])
    hz_success, hz_output = run_ros_command(['timeout', '2', 'rostopic', 'hz', topic_name], timeout=3.0)

    result = [
        f"Topic Summary: {topic_name}",
        "=" * 60,
        f"Type: {topic_type}",
        "",
        "Publishers/Subscribers:",
        info_output if info_success else "Unknown",
        "",
        "Publishing Rate:",
        hz_output if hz_success else "Could not measure",
        "",
        "Message Structure:",
        def_output if def_success else "Unknown"
    ]

    return "\n".join(result)


@mcp.tool()
async def read_topic(topic_name: str, count: int = 1, timeout: float = 5.0) -> str:
    """Read messages from a topic (automatically handles different data types).

    Args:
        topic_name: Full name of the topic
        count: Number of messages to read
        timeout: Timeout in seconds

    Returns formatted message data.
    """
    # Get topic type first
    type_success, type_output = run_ros_command(['rostopic', 'type', topic_name])
    topic_type = type_output.strip() if type_success else ""

    # Handle images specially
    if 'Image' in topic_type:
        return await analyze_image_topic(topic_name)

    # Handle point clouds specially
    if 'PointCloud' in topic_type:
        return await analyze_pointcloud_topic(topic_name)

    # For everything else, use sample_topic_data
    return await sample_topic_data(topic_name, count)


@mcp.tool()
async def get_topic_hz(topic_name: str, duration: float = 2.0) -> str:
    """Measure the publishing frequency (Hz) of a topic."""
    duration = min(duration, 10.0)
    success, output = run_ros_command(
        ['timeout', str(duration), 'rostopic', 'hz', topic_name],
        timeout=duration + 2.0
    )

    if not success or not output.strip():
        return f"Could not measure frequency for topic '{topic_name}'"

    lines = output.strip().split('\n')
    result = [f"Topic: {topic_name}", f"Duration: {duration:.2f}s", "=" * 60]
    result.extend([line for line in lines if line.strip()])

    return "\n".join(result)


@mcp.tool()
async def find_topics_by_type(message_type: str) -> str:
    """Find all topics that publish a specific message type."""
    success, output = run_ros_command(['rostopic', 'find', message_type])

    if not success:
        return f"Error finding topics: {output}"

    topics = [t.strip() for t in output.strip().split('\n') if t.strip()]

    if not topics:
        return f"No topics found with message type '{message_type}'"

    result = [f"Topics using type '{message_type}':", "=" * 60]
    result.extend([f"  {topic}" for topic in topics])
    result.append(f"\nTotal: {len(topics)} topic(s)")

    return "\n".join(result)


@mcp.tool()
async def get_message_definition(message_type: str) -> str:
    """Get the full message definition for a ROS message type."""
    success, output = run_ros_command(['rosmsg', 'show', message_type])

    if not success:
        return f"Error getting message definition: {output}"

    return f"Message Definition: {message_type}\n{'=' * 60}\n{output.strip()}"


def main():
    """Run the MCP server."""
    try:
        print("ROS1 MCP Server (Complete Edition) initialized")
        print(f"OpenCV available: {HAS_CV2}")
        print(f"Temp directory: {TEMP_DIR}")

        env_ok, env_msg = check_ros_environment()
        if not env_ok:
            print(f"WARNING: {env_msg}")

        mcp.run(transport='stdio')
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()