"""Subprocess and ROS command execution engine for NativaGPT.

Provides :class:`CommandExecution`, which runs arbitrary shell commands and
ROS1/ROS2 topic commands (``rostopic echo/info/hz``, ``ros2 topic
echo/info/hz``) as managed subprocesses or, where possible, via a fast
path that reads ROS topics directly instead of spawning a shell. Also
provides output-type detection (images, point clouds, structured data,
ROS topic payloads, JSON), heuristic error detection with human-readable
diagnosis and suggested fixes, a registry of detached background
processes, and lightweight in-memory performance metrics.
"""

import os
import sys
import time
import signal
import subprocess
import threading
import re
import json
from datetime import datetime
from collections import deque, defaultdict
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List, Callable, Tuple

try:
    import rospy
    import rostopic
except ImportError:
    rospy = None
    rostopic = None

from NativaGPT.lib.coloring_logger import logger


def timeit(func):
    """Decorator that measures and logs a function's execution time.

    Wraps ``func`` so each call is timed with ``time.time()``. The elapsed
    time in milliseconds is emitted via ``logger.debug`` after the wrapped
    function returns; the original return value is passed through
    unchanged.

    Args:
        func: The function to instrument.

    Returns:
        Callable: A wrapped version of ``func`` with the same signature
        and return value, plus a timing log side effect on each call.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = (time.time() - start) * 1000
        logger.debug(f"⏱️ {func.__name__}: {elapsed:.1f}ms")
        return result

    return wrapper


class CommandExecution:
    """Executes shell commands and ROS topic operations as managed subprocesses.

    Runs arbitrary shell commands via ``subprocess.Popen``, with background
    reader threads, adaptive polling, and early error detection so that
    long-running or failing commands can be surfaced quickly without
    blocking. Commands that are still running when polling stops are
    either registered in an internal process registry (for later output
    retrieval and stopping) or waited on to completion, depending on the
    caller's choice.

    ROS1 (``rostopic echo/info/hz``) and ROS2 (``ros2 topic
    echo/info/hz``) commands are detected via regex and, for ``echo``,
    served through a three-tier fast path (a pre-subscribed message
    cache, a cached topic configuration, or a temporary dynamic
    subscription) that talks to ``rospy``/``rostopic`` directly instead of
    spawning a subprocess. ROS node initialization is lazy and
    thread-safe, and topic types/configurations are cached to avoid
    repeated lookups.

    Command output is classified (image/point cloud/structured/CSV/rosbag
    file references, JSON, ROS topic payloads, or plain text) and, on
    failure, matched against a table of known ROS/system error patterns
    to produce a diagnosis with suggested fixes. A shared thread pool is
    used for parallel command execution and batched topic reads, and
    basic timing metrics are recorded for the main operations.
    """

    # ==================== OPTIMIZED REGEX PATTERNS ====================

    # More comprehensive ROS1 patterns
    _ROS1_ECHO_PATTERN = re.compile(
        r"rostopic\s+echo\s+([/\w\-]+)(?:\s|$)", re.IGNORECASE
    )
    _ROS1_INFO_PATTERN = re.compile(
        r"rostopic\s+info\s+([/\w\-]+)(?:\s|$)", re.IGNORECASE
    )
    _ROS1_HZ_PATTERN = re.compile(r"rostopic\s+hz\s+([/\w\-]+)(?:\s|$)", re.IGNORECASE)

    # More comprehensive ROS2 patterns
    _ROS2_ECHO_PATTERN = re.compile(
        r"ros2\s+topic\s+echo\s+([/\w\-]+)(?:\s|$)", re.IGNORECASE
    )
    _ROS2_INFO_PATTERN = re.compile(
        r"ros2\s+topic\s+info\s+([/\w\-]+)(?:\s|$)", re.IGNORECASE
    )
    _ROS2_HZ_PATTERN = re.compile(
        r"ros2\s+topic\s+hz\s+([/\w\-]+)(?:\s|$)", re.IGNORECASE
    )

    # Enhanced file patterns with more extensions
    _FILE_PATTERNS = {
        "image": re.compile(
            r"([\/\w\-\.]+\.(?:png|jpg|jpeg|gif|bmp|tiff?|webp|svg|ico))", re.IGNORECASE
        ),
        "pointcloud": re.compile(
            r"([\/\w\-\.]+\.(?:pcd|ply|xyz|pts|las|laz|e57))", re.IGNORECASE
        ),
        "structured": re.compile(
            r"([\/\w\-\.]+\.(?:json|yaml|yml|xml|toml))", re.IGNORECASE
        ),
        "csv": re.compile(r"([\/\w\-\.]+\.(?:csv|tsv|dat))", re.IGNORECASE),
        "rosbag": re.compile(r"([\/\w\-\.]+\.(?:bag|db3))", re.IGNORECASE),
    }

    # Enhanced error detection
    _ERROR_KEYWORDS = frozenset(
        [
            "error",
            "exception",
            "traceback",
            "fatal",
            "failed",
            "failure",
            "no such file",
            "cannot",
            "permission denied",
            "not found",
            "segmentation fault",
            "core dumped",
            "killed",
            "aborted",
            "warning",
            "critical",
            "panic",
            "invalid",
            "unable to",
        ]
    )

    # ROS-specific error patterns
    _ROS_ERROR_PATTERN = re.compile(
        r"\[(ERROR|FATAL|WARN)\]|roslaunch|Unable to communicate", re.IGNORECASE
    )

    # ==================== ERROR DIAGNOSIS & AUTO-FIX SYSTEM ====================

    # Comprehensive error patterns with diagnosis and fixes
    _ERROR_DIAGNOSIS_PATTERNS = [
        # ROS-specific errors
        {
            "pattern": re.compile(r"roscore.*not running", re.IGNORECASE),
            "category": "ROS_CORE_NOT_RUNNING",
            "diagnosis": "ROS master (roscore) is not running",
            "severity": "critical",
            "fixes": [
                "Start ROS master: roscore",
                "If using ROS 2: source your ROS 2 workspace and run: ros2 launch",
                "Check if ROS environment is sourced: echo $ROS_DISTRO",
            ],
        },
        {
            "pattern": re.compile(
                r"Unable to communicate with r\w+master", re.IGNORECASE
            ),
            "category": "ROS_MASTER_UNAVAILABLE",
            "diagnosis": "Cannot communicate with ROS master",
            "severity": "critical",
            "fixes": [
                "Ensure roscore is running in a terminal",
                "Check ROS_MASTER_URI environment variable: echo $ROS_MASTER_URI",
                "Verify network connectivity (if using remote master)",
                "Try: export ROS_MASTER_URI=http://localhost:11311",
            ],
        },
        {
            "pattern": re.compile(r"no such package", re.IGNORECASE),
            "category": "PACKAGE_NOT_FOUND",
            "diagnosis": "ROS package not found",
            "severity": "error",
            "fixes": [
                "Check if package is installed: rospack find <package_name>",
                "Build your workspace: cd ~/catkin_ws && catkin_make",
                "Source your workspace: source ~/catkin_ws/devel/setup.bash",
                "For ROS 2: cd ~/ros2_ws && colcon build && source install/setup.bash",
            ],
        },
        {
            "pattern": re.compile(r"cannot find.*launch", re.IGNORECASE),
            "category": "LAUNCH_FILE_NOT_FOUND",
            "diagnosis": "Launch file not found",
            "severity": "error",
            "fixes": [
                "Verify the launch file path is correct",
                "Check if package is built: catkin_make or colcon build",
                "List available launch files: roslaunch <package> --ros-args --list",
            ],
        },
        {
            "pattern": re.compile(r"connection refused", re.IGNORECASE),
            "category": "CONNECTION_REFUSED",
            "diagnosis": "Connection refused by node or service",
            "severity": "error",
            "fixes": [
                "Ensure the target node/service is running",
                "Check if nodes are initialized: rosnode list",
                "Verify service is available: rosservice list",
                "Wait for node to initialize before calling service",
            ],
        },
        {
            "pattern": re.compile(r"service.*not available", re.IGNORECASE),
            "category": "SERVICE_UNAVAILABLE",
            "diagnosis": "Required ROS service is not available",
            "severity": "error",
            "fixes": [
                "Check available services: rosservice list",
                "Ensure the service provider node is running",
                "For turtlesim: make sure turtlesim node is running first",
            ],
        },
        {
            "pattern": re.compile(r"topic.*not found", re.IGNORECASE),
            "category": "TOPIC_NOT_FOUND",
            "diagnosis": "ROS topic does not exist",
            "severity": "error",
            "fixes": [
                "List available topics: rostopic list",
                "Check topic spelling and namespace",
                "Ensure publisher is publishing to the topic",
                "For ROS 2: ros2 topic list",
            ],
        },
        {
            "pattern": re.compile(r"topic.*type mismatch", re.IGNORECASE),
            "category": "TOPIC_TYPE_MISMATCH",
            "diagnosis": "Topic message type does not match",
            "severity": "error",
            "fixes": [
                "Check topic type: rostopic type <topic>",
                "Use correct message type for publishing",
                "Verify the publisher and subscriber use same message type",
            ],
        },
        {
            "pattern": re.compile(r"parameter.*not found", re.IGNORECASE),
            "category": "PARAMETER_NOT_FOUND",
            "diagnosis": "ROS parameter not found",
            "severity": "warning",
            "fixes": [
                "List parameters: rosparam list",
                "Check parameter name and namespace",
                "Set parameter: rosparam set /param_name value",
            ],
        },
        # Turtlesim-specific errors
        {
            "pattern": re.compile(r"turtlesim.*not running", re.IGNORECASE),
            "category": "TURTLESIM_NOT_RUNNING",
            "diagnosis": "Turtlesim node is not running",
            "severity": "critical",
            "fixes": [
                "Start turtlesim: rosrun turtlesim turtlesim_node",
                "Or use launch file: roslaunch turtlesim neverest_simulation.launch",
                "For ROS 2: ros2 run turtlesim turtlesim_node",
            ],
        },
        # General system errors
        {
            "pattern": re.compile(r"no such file or directory", re.IGNORECASE),
            "category": "FILE_NOT_FOUND",
            "diagnosis": "File or directory does not exist",
            "severity": "error",
            "fixes": [
                "Check the file path is correct",
                "Verify the file exists: ls -la <path>",
                "Use absolute paths when possible",
                "Create the file if needed",
            ],
        },
        {
            "pattern": re.compile(r"permission denied", re.IGNORECASE),
            "category": "PERMISSION_DENIED",
            "diagnosis": "Permission denied to access file or resource",
            "severity": "error",
            "fixes": [
                "Check file permissions: ls -la <path>",
                "Add execute permission: chmod +x <file>",
                "Run with sudo if required (use caution)",
                "Check ownership: chown <user>:<group> <path>",
            ],
        },
        {
            "pattern": re.compile(r"command not found", re.IGNORECASE),
            "category": "COMMAND_NOT_FOUND",
            "diagnosis": "Command executable not found",
            "severity": "error",
            "fixes": [
                "Install the required package",
                "Check if binary is in PATH: which <command>",
                "Source the ROS workspace",
                "Verify package installation: dpkg -l | grep <package>",
            ],
        },
        {
            "pattern": re.compile(r"import error|module not found", re.IGNORECASE),
            "category": "PYTHON_IMPORT_ERROR",
            "diagnosis": "Python module import failed",
            "severity": "error",
            "fixes": [
                "Install required Python package: pip install <module>",
                "Install ROS Python package: sudo apt install ros-<distro>-<package>",
                "Check Python version compatibility",
                "Verify PYTHONPATH includes the module location",
            ],
        },
        {
            "pattern": re.compile(r"segmentation fault|core dumped", re.IGNORECASE),
            "category": "SEGMENTATION_FAULT",
            "diagnosis": "Process crashed with segmentation fault",
            "severity": "critical",
            "fixes": [
                "Check for memory issues or invalid pointers",
                "Review recent code changes",
                "Run with gdb for debugging: gdb --args <command>",
                "Check system resources: free -h, dmesg | tail",
            ],
        },
        {
            "pattern": re.compile(r"timeout", re.IGNORECASE),
            "category": "TIMEOUT",
            "diagnosis": "Operation timed out",
            "severity": "warning",
            "fixes": [
                "Increase timeout value",
                "Check network connectivity",
                "Verify the target service is responsive",
                "Reduce data size if processing large messages",
            ],
        },
        {
            "pattern": re.compile(r"environment variable.*not set", re.IGNORECASE),
            "category": "ENV_VAR_NOT_SET",
            "diagnosis": "Required environment variable is not set",
            "severity": "error",
            "fixes": [
                "Set the environment variable: export VAR_NAME=value",
                "Add to ~/.bashrc for persistence",
                "Common ROS vars: ROS_MASTER_URI, ROS_HOME, ROS_PACKAGE_PATH",
                "Source ROS setup: source /opt/ros/<distro>/setup.bash",
            ],
        },
        {
            "pattern": re.compile(r"address already in use", re.IGNORECASE),
            "category": "PORT_IN_USE",
            "diagnosis": "Network port already in use",
            "severity": "error",
            "fixes": [
                "Find process using port: lsof -i :<port>",
                "Kill the process: kill <PID>",
                "Use a different port",
                "Check for zombie processes",
            ],
        },
        {
            "pattern": re.compile(r"xmlrpc.*error|rosrpc.*error", re.IGNORECASE),
            "category": "XMLRPC_ERROR",
            "diagnosis": "XML-RPC communication error",
            "severity": "error",
            "fixes": [
                "Restart roscore",
                "Check network firewall settings",
                "Verify ROS_MASTER_URI is correct",
                "Ensure all nodes are on same network",
            ],
        },
    ]

    def __init__(self, topic_reader_handler=None):
        """Initializes the execution engine, thread pool, caches, and warmup.

        Args:
            topic_reader_handler: Optional handler (e.g. a
                ``TopicReaderHandler``) that supplies ROS topic
                subscription configuration and message conversion logic.
                When provided, its topic configuration is used to
                pre-build the topic cache and its pre-subscribed messages
                are used by the fast ROS topic read path. When ``None``,
                ROS topic reads fall back to dynamic, on-demand
                subscriptions without message conversion.
        """
        self._proc_registry: Dict[int, Dict[str, Any]] = {}
        self.topic_reader = topic_reader_handler

        # Thread pool with optimized worker count
        self.executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="cmd_exec")

        # Enhanced caching
        self._topic_config_cache: Dict[str, Dict[str, Any]] = {}
        self._topic_type_cache: Dict[str, Any] = {}  # NEW: Cache topic types
        self._cache_lock = threading.RLock()  # Reentrant lock

        # Lazy ROS initialization
        self._ros_initialized = False
        self._ros_init_lock = threading.Lock()

        # Performance metrics
        self._metrics = defaultdict(list)
        self._metrics_lock = threading.Lock()

        # Pre-build topic config cache if available
        if self.topic_reader:
            self._build_topic_cache()

        logger.info("CommandExecution v3.0 initialized (lazy ROS mode)")

        # Performance optimization: Pre-warm common command shells
        self._shell_warmed = False
        self._ros_env_checked = False
        self._start_warmup()

    def _start_warmup(self):
        """Starts a daemon thread that pre-warms subprocess spawning and checks the ROS environment."""

        def warmup():
            try:
                # Pre-spawn a shell to warm up subprocess
                import subprocess

                test_proc = subprocess.run(
                    ["echo", "warmup"], capture_output=True, timeout=1
                )
                self._shell_warmed = True

                # Quick ROS environment check
                if rospy:
                    import os

                    if os.environ.get("ROS_DISTRO"):
                        self._ros_env_checked = True

                logger.debug("✓ Command execution warmup complete")
            except Exception as e:
                logger.debug(f"Warmup skipped: {e}")

        warmup_thread = threading.Thread(target=warmup, daemon=True)
        warmup_thread.start()

    def _ensure_ros_initialized(self) -> bool:
        """Lazily and thread-safely initializes the ROS node on first use; returns whether ROS is available."""
        if self._ros_initialized:
            return True

        if not rospy:
            return False

        with self._ros_init_lock:
            if self._ros_initialized:  # Double-check
                return True

            try:
                if not rospy.core.is_initialized():
                    rospy.init_node(
                        "nativagpt_cmd_exec",
                        anonymous=True,
                        disable_signals=True,
                        log_level=rospy.ERROR,  # Reduce logging overhead
                    )
                    logger.info("ROS node initialized (lazy)")
                self._ros_initialized = True
                return True
            except Exception as e:
                logger.warning(f"ROS initialization failed: {e}")
                return False

    def _build_topic_cache(self):
        """Builds an in-memory cache of topic name to topic config from `topic_reader` for O(1) lookup."""
        try:
            if not self.topic_reader:
                return

            topic_config = self.topic_reader.topic_cfg.get("subscriptions", {})
            ros_topics = topic_config.get("ros_topics", [])

            with self._cache_lock:
                for topic in ros_topics:
                    topic_name = topic.get("name")
                    if topic_name:
                        self._topic_config_cache[topic_name] = topic

            logger.info(f"Cached {len(self._topic_config_cache)} topic configurations")
        except Exception as e:
            logger.warning(f"Failed to build topic cache: {e}")

    def _record_metric(self, operation: str, duration_ms: float):
        """Appends a timing sample for `operation`, keeping only the most recent 100 samples."""
        with self._metrics_lock:
            self._metrics[operation].append(duration_ms)
            # Keep only last 100 measurements
            if len(self._metrics[operation]) > 100:
                self._metrics[operation].pop(0)

    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Computes aggregate timing statistics for all recorded operations.

        Returns:
            Dict[str, Dict[str, float]]: A mapping from operation name (as
            passed to `_record_metric`) to a dict with `avg_ms`, `min_ms`,
            `max_ms`, and `count`, computed over the most recent samples
            recorded for that operation.
        """
        stats = {}
        with self._metrics_lock:
            for op, times in self._metrics.items():
                if times:
                    stats[op] = {
                        "avg_ms": sum(times) / len(times),
                        "min_ms": min(times),
                        "max_ms": max(times),
                        "count": len(times),
                    }
        return stats

    # ==================== ROS TOPIC DETECTION ====================

    def _parse_ros_topic_command(self, command: str) -> Optional[Dict[str, Any]]:
        """Detects ROS1/ROS2 `rostopic`/`ros2 topic` echo, info, or hz commands via regex.

        Returns:
            A dict with `is_ros_topic`, `ros_version`, `operation`,
            `topic_name`, and `once` if `command` matches a known
            pattern, otherwise `None`.
        """
        cmd_lower = command.lower().strip()

        # ROS1 echo
        match = self._ROS1_ECHO_PATTERN.search(cmd_lower)
        if match:
            return {
                "is_ros_topic": True,
                "ros_version": "ros1",
                "operation": "echo",
                "topic_name": match.group(1),
                "once": any(x in cmd_lower for x in ["-n 1", "-n1", "--once", "-n=1"]),
            }

        # ROS2 echo
        match = self._ROS2_ECHO_PATTERN.search(cmd_lower)
        if match:
            return {
                "is_ros_topic": True,
                "ros_version": "ros2",
                "operation": "echo",
                "topic_name": match.group(1),
                "once": "--once" in cmd_lower or "--times 1" in cmd_lower,
            }

        # ROS1 info
        match = self._ROS1_INFO_PATTERN.search(cmd_lower)
        if match:
            return {
                "is_ros_topic": True,
                "ros_version": "ros1",
                "operation": "info",
                "topic_name": match.group(1),
                "once": True,
            }

        # ROS2 info
        match = self._ROS2_INFO_PATTERN.search(cmd_lower)
        if match:
            return {
                "is_ros_topic": True,
                "ros_version": "ros2",
                "operation": "info",
                "topic_name": match.group(1),
                "once": True,
            }

        # ROS1 hz
        match = self._ROS1_HZ_PATTERN.search(cmd_lower)
        if match:
            return {
                "is_ros_topic": True,
                "ros_version": "ros1",
                "operation": "hz",
                "topic_name": match.group(1),
                "once": False,
            }

        # ROS2 hz
        match = self._ROS2_HZ_PATTERN.search(cmd_lower)
        if match:
            return {
                "is_ros_topic": True,
                "ros_version": "ros2",
                "operation": "hz",
                "topic_name": match.group(1),
                "once": False,
            }

        return None

    # ==================== CACHED TOPIC TYPE LOOKUP ====================

    @lru_cache(maxsize=128)
    def _get_topic_type_cached(self, topic_name: str) -> Optional[Any]:
        """Looks up and caches (via `lru_cache`) the ROS message class for a topic.

        Returns:
            The topic's message type class, or `None` if ROS isn't
            initialized or the topic has no publishers.
        """
        if not self._ensure_ros_initialized():
            return None

        try:
            topic_type, _, _ = rostopic.get_topic_class(topic_name, blocking=False)
            return topic_type
        except Exception as e:
            logger.debug(f"Failed to get topic type for {topic_name}: {e}")
            return None

    # ==================== OPTIMIZED DIRECT TOPIC READING ====================

    @timeit
    def _read_ros_topic_directly(self, topic_info: Dict[str, Any]) -> Dict[str, Any]:
        """Reads one message for a ROS topic using a three-tier fallback strategy.

        Tries, in order: (1) an already-subscribed message no older than
        10 seconds from `topic_reader`'s live cache, (2) a topic config
        cached from `topic_reader.topic_cfg`, delegating the actual read
        to `topic_reader._read_ros_topic`, and (3) a dynamic, on-demand
        subscription via `_read_ros_topic_dynamic`. Falls through to the
        next tier when a faster tier is unavailable or yields no data.

        Args:
            topic_info: Parsed topic command dict as returned by
                `_parse_ros_topic_command`; only `topic_name` is used.

        Returns:
            A dict describing the read result. On success it includes
            `success`, `topic_name`, `modality`, `data`, `message_type`,
            `timestamp`, `files`, and `source`. On failure it includes
            `success: False` and an `error` message.
        """
        if not self.topic_reader:
            return {"success": False, "error": "TopicReaderHandler not available"}

        topic_name = topic_info.get("topic_name")

        try:
            # Tier 1: Check if we have pre-subscribed data
            if self.topic_reader and hasattr(self.topic_reader, "_latest_ros_messages"):
                with self.topic_reader._ros_lock:
                    if topic_name in self.topic_reader._latest_ros_messages:
                        msg, timestamp = self.topic_reader._latest_ros_messages[
                            topic_name
                        ]

                        # Check if message is fresh enough (< 10 seconds old)
                        age = time.time() - timestamp
                        if age < 10.0:
                            logger.info(
                                f"Using pre-subscribed data for {topic_name} (age: {age:.1f}s)"
                            )

                            # Convert using existing logic
                            topic_config = self._topic_config_cache.get(topic_name, {})
                            msg_type = topic_config.get("message_type", "unknown")

                            modality, data, extra = (
                                self.topic_reader._convert_ros_message(msg_type, msg)
                            )

                            result = {
                                "success": True,
                                "topic_name": topic_name,
                                "modality": modality,
                                "data": data,
                                "message_type": msg_type,
                                "timestamp": datetime.fromtimestamp(
                                    timestamp
                                ).isoformat(),
                                "files": [],
                                "source": "pre_subscribed",
                            }

                            if (
                                modality == "image"
                                and isinstance(data, str)
                                and os.path.exists(data)
                            ):
                                result["files"].append(data)

                            return result

            # Tier 2: Check cache for topic config
            with self._cache_lock:
                matching_topic = self._topic_config_cache.get(topic_name)

            if matching_topic:
                logger.info(f"Using cached config for {topic_name}")
                topic_data = self.topic_reader._read_ros_topic(matching_topic)

                if not topic_data:
                    # Cache miss, fall through to dynamic reading
                    logger.info(f"No cached data, falling back to dynamic read")
                else:
                    result = {
                        "success": True,
                        "topic_name": topic_name,
                        "modality": topic_data.get("modality"),
                        "data": topic_data.get("data"),
                        "message_type": topic_data.get("extra", {}).get("message_type"),
                        "timestamp": topic_data.get("timestamp"),
                        "files": [],
                        "source": "cached_config",
                    }

                    if result["modality"] == "image":
                        data_path = topic_data.get("data")
                        if data_path and os.path.exists(data_path):
                            result["files"].append(data_path)

                    return result

            # Tier 3: Dynamic reading
            logger.info(f"No cache available for {topic_name}, using dynamic read...")
            return self._read_ros_topic_dynamic(topic_name, timeout=3.0)

        except Exception as e:
            logger.error(f"Error reading ROS topic {topic_name}: {e}")
            return {"success": False, "error": f"Failed to read topic: {str(e)}"}

    # ==================== ENHANCED DYNAMIC TOPIC READING ====================

    @timeit
    def _read_ros_topic_dynamic(
        self, topic_name: str, timeout: float = 2.0, max_wait: float = 5.0
    ) -> Dict[str, Any]:
        """Reads a single message from a ROS topic via a temporary subscription.

        Looks up the topic's message type (cached), subscribes with a
        callback that captures only the first message and signals a
        `threading.Event`, then waits up to `min(timeout, max_wait)`
        seconds for the event before always unregistering the
        subscriber. Records a `topic_read_timeout` or
        `topic_read_success` performance metric.

        Args:
            topic_name: Name of the ROS topic to read.
            timeout: Maximum seconds to wait for a message.
            max_wait: Hard upper bound on the wait time, used instead of
                `timeout` if smaller.

        Returns:
            A dict with `success` and, on success, `topic_name`,
            `modality`, `data`, `message_type`, `timestamp`, `files`,
            `source`, and `read_time_ms`. On failure, `success: False`
            and an `error` message.
        """
        if not self._ensure_ros_initialized():
            return {"success": False, "error": "rospy not available"}

        start_time = time.time()

        try:
            # Get topic type (cached)
            topic_type = self._get_topic_type_cached(topic_name)

            if topic_type is None:
                return {
                    "success": False,
                    "error": f"Topic {topic_name} not found or has no publishers",
                }

            logger.info(f"Dynamically reading {topic_name} (type: {topic_type._type})")

            # Use Event for instant timeout response
            received_msg = {"msg": None, "timestamp": None}
            msg_event = threading.Event()

            def callback(msg):
                if not msg_event.is_set():  # Only capture first message
                    received_msg["msg"] = msg
                    received_msg["timestamp"] = time.time()
                    msg_event.set()

            # Create temporary subscriber
            sub = rospy.Subscriber(topic_name, topic_type, callback, queue_size=1)

            try:
                # Wait with instant response on message arrival
                if not msg_event.wait(timeout=min(timeout, max_wait)):
                    elapsed = (time.time() - start_time) * 1000
                    self._record_metric("topic_read_timeout", elapsed)
                    return {
                        "success": False,
                        "error": f"Timeout waiting for message on {topic_name} ({timeout:.1f}s)",
                    }

                # Message received!
                elapsed = (time.time() - start_time) * 1000
                self._record_metric("topic_read_success", elapsed)
                logger.info(f"✓ Topic read in {elapsed:.1f}ms")

                # Convert the message
                if self.topic_reader:
                    msg_type_str = topic_type._type
                    modality, data, extra = self.topic_reader._convert_ros_message(
                        msg_type_str, received_msg["msg"]
                    )

                    result = {
                        "success": True,
                        "topic_name": topic_name,
                        "modality": modality,
                        "data": data,
                        "message_type": msg_type_str,
                        "timestamp": datetime.fromtimestamp(
                            received_msg["timestamp"]
                        ).isoformat(),
                        "files": [],
                        "source": "dynamic",
                        "read_time_ms": elapsed,
                    }

                    # Add image file if applicable
                    if (
                        modality == "image"
                        and isinstance(data, str)
                        and os.path.exists(data)
                    ):
                        result["files"].append(data)

                    return result
                else:
                    # Fallback without TopicReaderHandler
                    return {
                        "success": True,
                        "topic_name": topic_name,
                        "modality": "text",
                        "data": str(received_msg["msg"]),
                        "message_type": topic_type._type,
                        "timestamp": datetime.fromtimestamp(
                            received_msg["timestamp"]
                        ).isoformat(),
                        "files": [],
                        "source": "dynamic",
                        "read_time_ms": elapsed,
                    }

            finally:
                # Always unsubscribe (instant cleanup)
                try:
                    sub.unregister()
                except:
                    pass

        except Exception as e:
            logger.error(f"Error dynamically reading topic {topic_name}: {e}")
            import traceback

            traceback.print_exc()
            return {"success": False, "error": f"Failed to read topic: {str(e)}"}

    # ==================== BATCH TOPIC READING ====================

    def read_multiple_topics(
        self, topic_names: List[str], timeout: float = 2.0
    ) -> Dict[str, Dict[str, Any]]:
        """Reads multiple ROS topics concurrently using the shared thread pool.

        Args:
            topic_names: Names of the topics to read; each is read once
                via `_read_ros_topic_directly` (equivalent to `echo -n 1`).
            timeout: Extra seconds (beyond the number of topics) to wait
                for all reads to complete before giving up on the
                remaining futures.

        Returns:
            Dict[str, Dict[str, Any]]: A mapping from each topic name to
            its read result dict (same shape as
            `_read_ros_topic_directly`'s return value). Topics whose
            future raised or timed out get
            `{"success": False, "error": ...}`.
        """
        results = {}

        def read_single(topic_name):
            topic_info = {"topic_name": topic_name, "operation": "echo", "once": True}
            return topic_name, self._read_ros_topic_directly(topic_info)

        # Submit all reads in parallel
        futures = {
            self.executor.submit(read_single, name): name for name in topic_names
        }

        # Collect results
        for future in as_completed(futures, timeout=timeout + 1):
            try:
                topic_name, result = future.result()
                results[topic_name] = result
            except Exception as e:
                topic_name = futures[future]
                logger.error(f"Error reading {topic_name}: {e}")
                results[topic_name] = {"success": False, "error": str(e)}

        return results

    # ==================== ERROR DIAGNOSIS METHODS ====================

    def diagnose_error(
        self, stdout: str, stderr: str, command: str = ""
    ) -> Dict[str, Any]:
        """Matches command output against known error patterns and suggests fixes.

        Args:
            stdout: Captured standard output of the command.
            stderr: Captured standard error of the command.
            command: The original command string, included in the result
                for context.

        Returns:
            Dict[str, Any]: `{"has_error": False}` if no error was
            recognized (including when both `stdout` and `stderr` are
            blank). Otherwise a dict with `has_error: True`, `category`,
            `diagnosis`, `severity` (`critical`, `error`, or `warning`),
            `fixes` (list of suggested remediation steps),
            `original_error` (matching snippet, truncated to 200 chars),
            and `command`. Falls back to a generic `GENERIC_ERROR`
            diagnosis when `_default_error_match` detects an error that
            doesn't match any of the specific patterns.
        """
        combined = f"{stdout}\n{stderr}"
        if not combined.strip():
            return {"has_error": False}

        combined_lower = combined.lower()

        for error_info in self._ERROR_DIAGNOSIS_PATTERNS:
            if error_info["pattern"].search(combined_lower):
                # Find the matching line for context
                lines = combined.split("\n")
                matching_line = ""
                for line in lines:
                    if error_info["pattern"].search(line.lower()):
                        matching_line = line.strip()
                        break

                return {
                    "has_error": True,
                    "category": error_info["category"],
                    "diagnosis": error_info["diagnosis"],
                    "severity": error_info["severity"],
                    "fixes": error_info["fixes"],
                    "original_error": matching_line[:200]
                    if matching_line
                    else combined[:200],
                    "command": command,
                }

        # Generic error if no specific pattern matched
        if self._default_error_match(combined):
            return {
                "has_error": True,
                "category": "GENERIC_ERROR",
                "diagnosis": "Command failed with an error",
                "severity": "error",
                "fixes": [
                    "Check the error message above for details",
                    "Verify all required services are running",
                    "Review the command syntax",
                    "Check system logs for more information: dmesg | tail",
                ],
                "original_error": combined[:200],
                "command": command,
            }

        return {"has_error": False}

    def format_diagnosis_for_output(self, diagnosis: Dict[str, Any]) -> str:
        """Formats a `diagnose_error` result as a human-readable, emoji-prefixed message.

        Args:
            diagnosis: A diagnosis dict as returned by `diagnose_error`.

        Returns:
            A multi-line string with the diagnosis category, description,
            and numbered suggested fixes, or an empty string if
            `diagnosis` has no error (`has_error` is falsy).
        """
        if not diagnosis.get("has_error"):
            return ""

        severity_icons = {"critical": "🔴", "error": "🟠", "warning": "🟡"}

        icon = severity_icons.get(diagnosis.get("severity", "error"), "❌")

        lines = [
            f"{icon} **ERROR DIAGNOSIS**",
            f"Category: {diagnosis.get('category', 'UNKNOWN')}",
            f"Diagnosis: {diagnosis.get('diagnosis', 'Unknown error')}",
            "",
            "**Suggested Fixes:**",
        ]

        for i, fix in enumerate(diagnosis.get("fixes", []), 1):
            lines.append(f"  {i}. {fix}")

        if diagnosis.get("original_error"):
            lines.extend(
                ["", f"Error snippet: _{diagnosis.get('original_error', '')}_"]
            )

        return "\n".join(lines)

    # ==================== ENHANCED ERROR DETECTION ====================

    def _default_error_match(self, s: str) -> bool:
        """Checks whether `s` contains a known error keyword or matches the ROS error regex."""
        if not s:
            return False

        s_lower = s.lower()

        # Quick keyword check
        if any(keyword in s_lower for keyword in self._ERROR_KEYWORDS):
            return True

        # ROS-specific error pattern
        if self._ROS_ERROR_PATTERN.search(s):
            return True

        return False

    # ==================== OPTIMIZED OUTPUT DETECTION ====================

    @timeit
    def _detect_output_type(
        self, stdout: str, stderr: str, command: str = ""
    ) -> Dict[str, Any]:
        """Classifies command output and extracts structured data, files, or ROS topic payloads.

        Checks for a stderr-based error first, then a ROS topic echo fast
        path via `_read_ros_topic_directly`, then scans the combined
        stdout/stderr for file paths of known types (image, point cloud,
        structured, CSV, rosbag) — in parallel via the thread pool for
        larger outputs — and finally attempts to parse stdout as JSON
        (including fenced ```json``` blocks).

        Args:
            stdout: Captured standard output.
            stderr: Captured standard error.
            command: The original command string, used to detect ROS
                topic commands.

        Returns:
            A dict with `type` (e.g. `text`, `error`, `json`, `image`, or
            a ROS modality), `files` (existing file paths found in the
            output), `data`, `has_error`, and `ros_topic_data` (populated
            when a ROS topic read succeeded).
        """
        output_info = {
            "type": "text",
            "files": [],
            "data": stdout,
            "has_error": bool(stderr.strip()),
            "ros_topic_data": None,
        }

        # Fast error check
        if stderr.strip() and self._default_error_match(stderr):
            output_info["type"] = "error"
            output_info["data"] = stderr
            return output_info

        # ROS topic command check (fast path)
        ros_topic_info = self._parse_ros_topic_command(command)
        if ros_topic_info and ros_topic_info["operation"] == "echo":
            topic_data = self._read_ros_topic_directly(ros_topic_info)

            if topic_data.get("success"):
                output_info.update(
                    {
                        "ros_topic_data": topic_data,
                        "type": topic_data.get("modality", "text"),
                        "data": topic_data.get("data"),
                        "files": topic_data.get("files", []),
                        "topic_name": topic_data.get("topic_name"),
                        "message_type": topic_data.get("message_type"),
                        "timestamp": topic_data.get("timestamp"),
                        "source": topic_data.get("source", "unknown"),
                    }
                )
                return output_info
            else:
                logger.warning(f"Failed to read topic: {topic_data.get('error')}")

        # File extraction with parallel scanning
        combined = stdout + "\n" + stderr

        # Parallel file pattern matching
        def scan_pattern(item):
            file_type, pattern = item
            matches = pattern.findall(combined)
            return [(match, file_type) for match in matches if os.path.exists(match)]

        found_files = []
        file_types = []

        if len(combined) > 100:  # Only parallelize for larger outputs
            futures = [
                self.executor.submit(scan_pattern, item)
                for item in self._FILE_PATTERNS.items()
            ]

            for future in as_completed(futures):
                try:
                    results = future.result()
                    for path, ftype in results:
                        found_files.append(path)
                        file_types.append(ftype)
                except:
                    pass
        else:
            # Sequential for small outputs
            for file_type, pattern in self._FILE_PATTERNS.items():
                matches = pattern.findall(combined)
                for match in matches:
                    if os.path.exists(match):
                        found_files.append(match)
                        file_types.append(file_type)

        output_info["files"] = list(set(found_files))

        # Type determination
        if file_types:
            output_info["type"] = file_types[0]

        # Enhanced JSON parsing
        if not found_files and not ros_topic_info:
            stripped = stdout.strip()
            if stripped and (stripped[0] in "{[" or stripped.startswith("```json")):
                try:
                    # Handle code blocks
                    if stripped.startswith("```json"):
                        stripped = stripped.split("```json")[1].split("```")[0].strip()
                    elif stripped.startswith("```"):
                        stripped = stripped.split("```")[1].split("```")[0].strip()

                    json_data = json.loads(stripped)
                    output_info["type"] = "json"
                    output_info["data"] = json_data
                except:
                    pass

        return output_info

    # ==================== OPTIMIZED BUFFER OPERATIONS ====================

    def _reader_thread(self, pipe, buffer: deque, name: str):
        """Continuously reads lines from `pipe` into `buffer` until the pipe closes, then closes it."""
        try:
            # Read in chunks for better performance
            while True:
                line = pipe.readline()
                if not line:
                    break
                buffer.append(line)
        except (IOError, ValueError):
            # Expected errors when pipe closes
            pass
        except Exception as e:
            logger.debug(f"{name} reader thread error: {e}")
        finally:
            try:
                pipe.close()
            except:
                pass

    def _join_buffer_efficient(self, buffer: deque) -> str:
        """Joins a deque (or list) of buffered output lines into a single stripped string."""
        if not buffer:
            return ""
        return "".join(buffer).strip()

    # ==================== MAIN EXECUTION API ====================

    def execute_command(
        self,
        command: str,
        wait_for_errors_seconds: float = 3.0,
        timeout: Optional[float] = None,
        error_match: Optional[Callable[[str], bool]] = None,
        detach_on_no_error: bool = True,
    ) -> Dict[str, Any]:
        """Executes a shell command, fast-pathing ROS topic echo reads.

        For `rostopic echo`/`ros2 topic echo` commands, attempts a direct
        ROS read via `_read_ros_topic_directly` first; only on failure
        does it fall back to spawning a subprocess. Otherwise (or after a
        fast-path failure), runs `command` in a shell subprocess with
        background reader threads draining stdout/stderr into bounded
        deques, and polls with adaptive, exponentially-backed-off sleeps.

        While waiting, the command may exit in time (returning its final
        result), have an error detected early in its stderr tail
        (returning with `running: True` and
        `note: "error_detected_early"`), or reach `timeout` /
        `wait_for_errors_seconds` without finishing — in which case it is
        either registered in `self._proc_registry` for later
        polling/stopping (`detach_on_no_error=True`) or waited on to
        completion (`detach_on_no_error=False`, which blocks).

        Args:
            command: The shell command to execute.
            wait_for_errors_seconds: How long to actively poll for an
                early error signal (or completion) before giving up on
                error detection.
            timeout: Optional hard deadline in seconds after which
                polling stops even if the process hasn't exited and no
                error was seen.
            error_match: Optional predicate called on the tail of stderr
                to detect an error early; defaults to
                `_default_error_match`.
            detach_on_no_error: If True (default) and the process is
                still running with no error seen when polling stops,
                register it as a detached background process instead of
                blocking further. If False, block until the process
                exits.

        Returns:
            Dict[str, Any]: Always includes `command`, `returncode`,
            `stdout`, `stderr`, `running`, `pid`, `note`, `output_info`
            (from `_detect_output_type`, possibly with a `diagnosis`
            key), and `execution_time_ms`.
        """
        start_time = time.time()
        logger.info(f"Executing: {command[:80]}...")

        # Fast path for ROS topic reads - ALWAYS USE FOR ECHO COMMANDS
        ros_topic_info = self._parse_ros_topic_command(command)
        if ros_topic_info and ros_topic_info["operation"] == "echo":
            # For echo commands, ALWAYS try direct read first (get one message)
            # This is much faster than subprocess
            logger.info(
                f"Using fast path for topic read: {ros_topic_info['topic_name']}"
            )
            topic_data = self._read_ros_topic_directly(ros_topic_info)

            if topic_data.get("success"):
                elapsed = (time.time() - start_time) * 1000
                self._record_metric("command_ros_topic", elapsed)

                logger.info(
                    f"✓ Direct topic read: {ros_topic_info['topic_name']} "
                    f"({elapsed:.1f}ms, source: {topic_data.get('source', 'unknown')})"
                )

                output_info = {
                    "type": topic_data.get("modality", "text"),
                    "files": topic_data.get("files", []),
                    "data": topic_data.get("data"),
                    "ros_topic_data": topic_data,
                    "topic_name": topic_data.get("topic_name"),
                    "message_type": topic_data.get("message_type"),
                    "has_error": False,
                }

                return {
                    "command": command,
                    "returncode": 0,
                    "stdout": f"Topic data retrieved: {ros_topic_info['topic_name']}",
                    "stderr": "",
                    "running": False,
                    "pid": None,
                    "note": "ros_topic_direct_read",
                    "output_info": output_info,
                    "execution_time_ms": elapsed,
                }
            else:
                # Direct read failed, log but continue to subprocess fallback
                logger.warning(
                    f"Direct topic read failed for {ros_topic_info['topic_name']}: "
                    f"{topic_data.get('error', 'unknown error')}, falling back to subprocess"
                )

        # Standard subprocess execution
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=8192,
            universal_newlines=True,
            preexec_fn=os.setsid,
        )

        # Adaptive buffer sizing based on command type
        max_buffer = 8000 if "rosbag" in command.lower() else 4000
        stdout_buf = deque(maxlen=max_buffer)
        stderr_buf = deque(maxlen=max_buffer)

        # Start reader threads
        t_out = threading.Thread(
            target=self._reader_thread,
            args=(proc.stdout, stdout_buf, "STDOUT"),
            daemon=True,
        )
        t_err = threading.Thread(
            target=self._reader_thread,
            args=(proc.stderr, stderr_buf, "STDERR"),
            daemon=True,
        )
        t_out.start()
        t_err.start()

        match_fn = error_match or self._default_error_match
        detected_error = False

        deadline = (
            time.time() + wait_for_errors_seconds if wait_for_errors_seconds else None
        )
        exit_deadline = time.time() + timeout if timeout else None

        # Adaptive polling with exponential backoff
        poll_interval = 0.02
        max_poll_interval = 0.1
        last_check = time.time()

        while True:
            rc = proc.poll()
            if rc is not None:
                # Process completed
                t_out.join(timeout=0.3)
                t_err.join(timeout=0.3)

                stdout_text = self._join_buffer_efficient(stdout_buf)
                stderr_text = self._join_buffer_efficient(stderr_buf)
                output_info = self._detect_output_type(
                    stdout_text, stderr_text, command
                )

                # Add error diagnosis
                if rc != 0 or stderr_text.strip():
                    diagnosis = self.diagnose_error(stdout_text, stderr_text, command)
                    output_info["diagnosis"] = diagnosis
                    if diagnosis.get("has_error"):
                        diagnosis_msg = self.format_diagnosis_for_output(diagnosis)
                        logger.warning(f"\n{diagnosis_msg}")

                elapsed = (time.time() - start_time) * 1000
                self._record_metric("command_completed", elapsed)

                logger.info(
                    f"✓ Command completed: rc={rc}, type={output_info.get('type')}, "
                    f"time={elapsed:.1f}ms"
                )

                return {
                    "command": command,
                    "returncode": rc,
                    "stdout": stdout_text,
                    "stderr": stderr_text,
                    "running": False,
                    "pid": None,
                    "note": None,
                    "output_info": output_info,
                    "execution_time_ms": elapsed,
                }

            # Check timeouts
            current_time = time.time()
            if exit_deadline and current_time >= exit_deadline:
                break

            # Error detection with adaptive checking
            if current_time - last_check >= poll_interval:
                tail_err = "".join(list(stderr_buf)[-10:])
                if tail_err and match_fn(tail_err):
                    detected_error = True
                    break
                last_check = current_time

                # Exponential backoff for polling
                poll_interval = min(poll_interval * 1.2, max_poll_interval)

            if deadline and current_time >= deadline:
                break

            time.sleep(poll_interval)

        # Still running - prepare for detachment or return
        stdout_preview = self._join_buffer_efficient(list(stdout_buf)[-400:])
        stderr_preview = self._join_buffer_efficient(list(stderr_buf)[-400:])
        output_info = self._detect_output_type(stdout_preview, stderr_preview, command)

        # Add error diagnosis for early detection
        if detected_error or stderr_preview.strip():
            diagnosis = self.diagnose_error(stdout_preview, stderr_preview, command)
            output_info["diagnosis"] = diagnosis
            if diagnosis.get("has_error"):
                diagnosis_msg = self.format_diagnosis_for_output(diagnosis)
                logger.warning(f"\n{diagnosis_msg}")

        elapsed = (time.time() - start_time) * 1000

        if detected_error:
            self._record_metric("command_error_detected", elapsed)
            logger.error(f"⚠️ Early error detected (time={elapsed:.1f}ms)")
            return {
                "command": command,
                "returncode": None,
                "stdout": stdout_preview,
                "stderr": stderr_preview,
                "running": True,
                "pid": proc.pid,
                "note": "error_detected_early",
                "output_info": output_info,
                "execution_time_ms": elapsed,
            }

        if detach_on_no_error:
            self._proc_registry[proc.pid] = {
                "process": proc,
                "stdout_buf": stdout_buf,
                "stderr_buf": stderr_buf,
                "command": command,
                "start_time": time.time(),
            }
            self._record_metric("command_detached", elapsed)
            logger.info(f"⏸️ Detached command: PID={proc.pid} (time={elapsed:.1f}ms)")
            return {
                "command": command,
                "returncode": None,
                "stdout": stdout_preview,
                "stderr": stderr_preview,
                "running": True,
                "pid": proc.pid,
                "note": "detached_no_errors_seen",
                "output_info": output_info,
                "execution_time_ms": elapsed,
            }
        else:
            logger.info("⏳ Waiting for command completion...")
            proc.wait()
            t_out.join(timeout=0.5)
            t_err.join(timeout=0.5)

            stdout_text = self._join_buffer_efficient(stdout_buf)
            stderr_text = self._join_buffer_efficient(stderr_buf)
            output_info = self._detect_output_type(stdout_text, stderr_text, command)

            elapsed = (time.time() - start_time) * 1000
            self._record_metric("command_waited", elapsed)

            return {
                "command": command,
                "returncode": proc.returncode,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "running": False,
                "pid": None,
                "note": None,
                "output_info": output_info,
                "execution_time_ms": elapsed,
            }

    # ==================== PROCESS MANAGEMENT ====================

    def stop_process(self, pid: int, grace_seconds: float = 2.0) -> Dict[str, Any]:
        """Stops a previously detached process, escalating from SIGTERM to SIGKILL.

        Sends `SIGTERM` to the process group and waits up to
        `grace_seconds` for it to exit (polled via a background thread
        and a `threading.Event` for prompt response); if it hasn't exited
        by then, sends `SIGKILL`. Removes the process from the internal
        registry regardless of the outcome.

        Args:
            pid: PID of a process previously registered via
                `execute_command` (i.e. present in
                `self._proc_registry`).
            grace_seconds: How long to wait after `SIGTERM` before
                escalating to `SIGKILL`.

        Returns:
            Dict[str, Any]: `{"ok": False, "error": ...}` if `pid` isn't
            registered. Otherwise `{"ok": True, ...}` with
            `already_stopped: True` if it had already exited, or
            `returncode` (which may be `None` if the process could not be
            confirmed stopped) otherwise.
        """
        info = self._proc_registry.get(pid)
        if not info:
            return {"ok": False, "error": f"PID {pid} not found"}

        proc = info["process"]
        if proc.poll() is not None:
            return {"ok": True, "already_stopped": True, "returncode": proc.returncode}

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception as e:
            logger.warning(f"SIGTERM failed for {pid}: {e}")

        # Event-based waiting for instant response
        stop_event = threading.Event()

        def wait_for_stop():
            while proc.poll() is None:
                time.sleep(0.01)
            stop_event.set()

        waiter = threading.Thread(target=wait_for_stop, daemon=True)
        waiter.start()

        # Wait with timeout
        stopped = stop_event.wait(timeout=grace_seconds)

        if not stopped:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                time.sleep(0.05)  # Brief wait after SIGKILL
            except Exception as e:
                logger.error(f"SIGKILL failed for {pid}: {e}")

        rc = proc.poll()
        self._proc_registry.pop(pid, None)
        return {"ok": True, "returncode": rc}

    def get_process_output(self, pid: int, tail_lines: int = 200) -> Dict[str, Any]:
        """Retrieves buffered output for a registered (detached) process.

        Args:
            pid: PID of a process previously registered via
                `execute_command`.
            tail_lines: Maximum number of most-recent buffered lines to
                return for each of stdout/stderr.

        Returns:
            Dict[str, Any]: `{"error": ...}` if `pid` isn't registered.
            Otherwise `pid`, `running`, `stdout_tail`, `stderr_tail`, and
            `returncode` (`None` while still running).
        """
        info = self._proc_registry.get(pid)
        if not info:
            return {"error": f"PID {pid} not found"}

        proc = info["process"]
        running = proc.poll() is None

        stdout_tail = self._join_buffer_efficient(
            list(info["stdout_buf"])[-tail_lines:]
        )
        stderr_tail = self._join_buffer_efficient(
            list(info["stderr_buf"])[-tail_lines:]
        )

        return {
            "pid": pid,
            "running": running,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "returncode": None if running else proc.returncode,
        }

    # ==================== BATCH EXECUTION ====================

    def get_commands_output(
        self,
        commands: List[str],
        wait_for_errors_seconds: float = 3.0,
        timeout: Optional[float] = None,
        detach_on_no_error: bool = True,
        error_match: Optional[Callable[[str], bool]] = None,
        parallel: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Executes multiple commands, batching one-shot ROS topic echo reads.

        Splits `commands` into one-shot `rostopic`/`ros2 topic echo`
        reads (read in parallel via `read_multiple_topics`) and all other
        commands (run via `_execute_commands_sequential` or
        `_execute_commands_parallel`, depending on `parallel`).

        Args:
            commands: Commands to execute.
            wait_for_errors_seconds: Passed through to `execute_command`
                for non-topic commands.
            timeout: Passed through to `execute_command` for non-topic
                commands.
            detach_on_no_error: Passed through to `execute_command` for
                non-topic commands.
            error_match: Passed through to `execute_command` for
                non-topic commands.
            parallel: If True, run non-topic commands concurrently via
                the thread pool; if False, run them sequentially.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A mapping from each original
            command string to a list containing its single result dict.
            Batched ROS topic reads are instead keyed by a synthesized
            `"rostopic echo <topic> -n 1"` string rather than the
            original command text.
        """
        # Optimize: if all commands are ROS topic reads, use batch reading
        ros_topics = []
        other_commands = []

        for cmd in commands:
            topic_info = self._parse_ros_topic_command(cmd)
            if topic_info and topic_info["operation"] == "echo" and topic_info["once"]:
                ros_topics.append(topic_info["topic_name"])
            else:
                other_commands.append(cmd)

        results = {}

        # Batch read ROS topics
        if ros_topics:
            logger.info(f"Batch reading {len(ros_topics)} ROS topics in parallel")
            topic_results = self.read_multiple_topics(ros_topics, timeout=3.0)

            for topic_name, topic_data in topic_results.items():
                cmd = f"rostopic echo {topic_name} -n 1"
                results[cmd] = [
                    {
                        "command": cmd,
                        "returncode": 0 if topic_data.get("success") else 1,
                        "stdout": f"Topic data retrieved: {topic_name}",
                        "stderr": ""
                        if topic_data.get("success")
                        else topic_data.get("error", ""),
                        "running": False,
                        "pid": None,
                        "note": "ros_topic_batch_read",
                        "output_info": {
                            "type": topic_data.get("modality", "text"),
                            "files": topic_data.get("files", []),
                            "data": topic_data.get("data"),
                            "ros_topic_data": topic_data,
                            "topic_name": topic_name,
                            "message_type": topic_data.get("message_type"),
                            "has_error": not topic_data.get("success"),
                        },
                    }
                ]

        # Execute other commands
        if other_commands:
            if parallel:
                other_results = self._execute_commands_parallel(
                    other_commands,
                    wait_for_errors_seconds,
                    timeout,
                    detach_on_no_error,
                    error_match,
                )
            else:
                other_results = self._execute_commands_sequential(
                    other_commands,
                    wait_for_errors_seconds,
                    timeout,
                    detach_on_no_error,
                    error_match,
                )
            results.update(other_results)

        return results

    def _execute_commands_sequential(
        self,
        commands: List[str],
        wait_for_errors_seconds: float,
        timeout: Optional[float],
        detach_on_no_error: bool,
        error_match: Optional[Callable[[str], bool]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Runs `commands` one at a time via `execute_command`, logging failures and ROS topic reads."""
        execute_result = {}

        for command in commands:
            try:
                if command not in execute_result:
                    execute_result[command] = []

                result = self.execute_command(
                    command=command,
                    wait_for_errors_seconds=wait_for_errors_seconds,
                    timeout=timeout,
                    error_match=error_match,
                    detach_on_no_error=detach_on_no_error,
                )

                execute_result[command].append(result)

                # Enhanced logging
                if result.get("returncode", 0) != 0 and not result.get("running"):
                    logger.error(f"❌ Command failed: {command[:50]}...")

                output_info = result.get("output_info", {})
                if output_info.get("ros_topic_data"):
                    logger.info(
                        f"✓ ROS topic: {output_info.get('topic_name')} -> "
                        f"{output_info.get('type')} "
                        f"({result.get('execution_time_ms', 0):.1f}ms)"
                    )

            except Exception as e:
                logger.error(f"❌ Error executing {command[:50]}...: {e}")

        return execute_result

    def _execute_commands_parallel(
        self,
        commands: List[str],
        wait_for_errors_seconds: float,
        timeout: Optional[float],
        detach_on_no_error: bool,
        error_match: Optional[Callable[[str], bool]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Runs `commands` concurrently via the thread pool and `execute_command`.

        WARNING: Only use for independent commands! Concurrent execution
        gives no ordering guarantees between commands.
        """
        execute_result = {}

        def execute_single(cmd):
            return cmd, self.execute_command(
                command=cmd,
                wait_for_errors_seconds=wait_for_errors_seconds,
                timeout=timeout,
                error_match=error_match,
                detach_on_no_error=detach_on_no_error,
            )

        # Submit all commands
        futures = {self.executor.submit(execute_single, cmd): cmd for cmd in commands}

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                command, result = future.result()
                if command not in execute_result:
                    execute_result[command] = []
                execute_result[command].append(result)

                if result.get("returncode", 0) != 0 and not result.get("running"):
                    logger.error(f"❌ Command failed: {command[:50]}...")

            except Exception as e:
                cmd = futures[future]
                logger.error(f"❌ Error in parallel execution of {cmd[:50]}...: {e}")

        return execute_result

    # ==================== CLEANUP ====================

    def __del__(self):
        """Logs final performance stats and best-effort cleans up the executor and tracked subprocesses.

        Runs on garbage collection: logs aggregate timing stats, shuts
        down `self.executor` without waiting, and attempts to `SIGKILL`
        any processes still present in `self._proc_registry`. All
        exceptions are swallowed since this runs during
        interpreter/object teardown.
        """
        try:
            # Log performance stats before cleanup
            stats = self.get_performance_stats()
            if stats:
                logger.info("📊 Performance Statistics:")
                for op, metrics in stats.items():
                    logger.info(
                        f"  {op}: avg={metrics['avg_ms']:.1f}ms, "
                        f"min={metrics['min_ms']:.1f}ms, "
                        f"max={metrics['max_ms']:.1f}ms, "
                        f"count={metrics['count']}"
                    )

            # Shutdown executor
            self.executor.shutdown(wait=False)

            # Kill any remaining processes
            for pid, info in list(self._proc_registry.items()):
                try:
                    proc = info["process"]
                    if proc.poll() is None:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except:
                    pass
        except:
            pass
