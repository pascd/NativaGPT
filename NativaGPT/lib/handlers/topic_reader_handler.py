import os
import glob
import time
import json
import base64
import queue
import shutil
import threading
import tempfile
import importlib
import pathlib
from collections import deque
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

try:
    import rospy
except Exception:
    rospy = None

try:
    from cv_bridge import CvBridge
except Exception:
    CvBridge = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None

try:
    from PIL import Image
except Exception:
    Image = None

from NativaGPT.lib.config_manager import ConfigManager
from NativaGPT.lib.coloring_logger import logger


class TopicReaderHandler:
    """
    TopicReaderHandler v2.3 - With Manual Toggle Control

    Key features:
    - Added topic_processing_enabled flag for runtime control
    - Can toggle topic processing on/off per request
    - Supports both persistent and JIT subscriptions
    """

    # Pre-compiled image magic bytes patterns
    _IMAGE_SIGNATURES = {
        b"\x89PNG": ".png",
        b"\xff\xd8": ".jpg",
        b"BM": ".bmp",
        b"GI": ".gif",
    }

    # IMAGE OPTIMIZATION SETTINGS
    MAX_IMAGE_DIMENSION = 1024  # Max width or height
    JPEG_QUALITY = 75  # Reduced from 90 for smaller files
    USE_FAST_JPEG = True  # Use faster JPEG encoding
    ENABLE_IMAGE_CACHE = True  # Cache converted images

    def __init__(self, config: ConfigManager):
        logger.info("Initializing TopicReaderHandler v2.3 (Toggle-Capable)...")

        self.config = config
        self.topic_cfg = self.config["topic_config"]
        self.max_hist = int(self.topic_cfg.get("max_message_history", 100))
        self.tmp_root = pathlib.Path(tempfile.mkdtemp(prefix="nativagpt_topics_"))

        # Configuration flags
        self.persistent_subs_enabled = self.topic_cfg.get("enable_persistent_subscriptions", False)
        self.auto_process_topics = self.topic_cfg.get("auto_process_topics", False)

        # Runtime toggle - can be changed dynamically
        self.topic_processing_enabled = self.auto_process_topics

        # Lazy initialization of cv_bridge
        self._bridge = None
        self._bridge_lock = threading.Lock()

        # Image cache for recently converted images
        self._image_cache: Dict[str, Tuple[str, float]] = {}
        self._image_cache_lock = threading.Lock()

        # Optimized history with deque (faster append/pop)
        self._history: deque = deque(maxlen=self.max_hist)
        self._history_lock = threading.Lock()

        # ROS
        self.ros_ok = self._check_ros_master()
        self._ros_subscribers: Dict[str, Any] = {}
        self._latest_ros_messages: Dict[str, Tuple[Any, float]] = {}
        self._ros_lock = threading.RLock()

        # MQTT
        self._mqtt_clients: Dict[Tuple[str, int], mqtt.Client] = {}
        self._mqtt_lock = threading.Lock()

        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="topic_reader")

        # Cache for topic configurations
        self._topic_cache: Dict[str, Dict[str, Any]] = {}
        self._build_topic_cache()

        # Only pre-subscribe if configured
        if self.ros_ok and self.persistent_subs_enabled:
            logger.info("[ROS] Persistent (pre-subscription) mode ENABLED.")
            self._init_ros_subscribers()
            self._wait_for_initial_messages(timeout=2.0)
        elif self.ros_ok:
            logger.info("[ROS] Persistent (pre-subscription) mode DISABLED. Using On-Demand (JIT) reads.")

        logger.info(f"TopicReaderHandler initialized. ROS: {self.ros_ok}, Auto-process: {self.auto_process_topics}")

    def enable_topic_processing(self):
        """Enable topic processing for the next request."""
        self.topic_processing_enabled = True
        logger.info("Topic processing ENABLED")

    def disable_topic_processing(self):
        """Disable topic processing for the next request."""
        self.topic_processing_enabled = False
        logger.info("Topic processing DISABLED")

    def toggle_topic_processing(self) -> bool:
        """Toggle topic processing and return new state."""
        self.topic_processing_enabled = not self.topic_processing_enabled
        logger.info(f"Topic processing toggled to: {self.topic_processing_enabled}")
        return self.topic_processing_enabled

    def is_topic_processing_enabled(self) -> bool:
        """Check if topic processing is currently enabled."""
        return self.topic_processing_enabled

    @property
    def bridge(self):
        """Lazy initialization of CvBridge."""
        if self._bridge is None and CvBridge is not None:
            with self._bridge_lock:
                if self._bridge is None:
                    self._bridge = CvBridge()
        return self._bridge

    def _build_topic_cache(self):
        """Pre-build cache of topic configurations."""
        subs = self.topic_cfg.get("subscriptions", {})

        for topic_list in [subs.get("ros_topics", []),
                           subs.get("mqtt_topics", []),
                           subs.get("file_topics", [])]:
            for topic in topic_list:
                if topic.get("enabled", False):
                    name = topic.get("name")
                    if name:
                        self._topic_cache[name] = topic

    def _check_ros_master(self) -> bool:
        """Check if ROS master is available and initialize node."""
        try:
            import rosgraph
            rosgraph.Master('/nativagpt_probe').getPid()

            if not rospy.core.is_initialized():
                rospy.init_node('nativagpt_topic_reader', anonymous=True, disable_signals=True)
                logger.info("ROS node initialized")

            logger.info("ROS Master reachable")
            return True
        except Exception as e:
            logger.debug(f"ROS Master not reachable: {e}")
            return False

    @lru_cache(maxsize=32)
    def _get_message_class(self, msg_type_str: str):
        """Cached message class loading."""
        pkg, cls = msg_type_str.split("/")
        return getattr(importlib.import_module(f"{pkg}.msg"), cls)

    def _init_ros_subscribers(self):
        """Create persistent subscribers for enabled ROS topics."""
        for t in self.topic_cfg.get("subscriptions", {}).get("ros_topics", []):
            if not t.get("enabled", False):
                continue

            name, msg_type_str = t.get("name"), t.get("message_type")
            if not (name and msg_type_str):
                continue

            try:
                msg_cls = self._get_message_class(msg_type_str)

                def callback(msg, topic_name=name):
                    with self._ros_lock:
                        self._latest_ros_messages[topic_name] = (msg, time.time())

                sub = rospy.Subscriber(name, msg_cls, callback, queue_size=1)
                self._ros_subscribers[name] = sub
                logger.info(f"[ROS] Subscribed to {name} ({msg_type_str})")
            except Exception as e:
                logger.error(f"[ROS] Failed to subscribe to {name}: {e}")

    def _wait_for_initial_messages(self, timeout: float):
        """Wait briefly for initial messages on all subscribed topics."""
        if not self._ros_subscribers:
            return

        start = time.time()
        expected = set(self._ros_subscribers.keys())

        while (time.time() - start) < timeout:
            with self._ros_lock:
                if set(self._latest_ros_messages.keys()) >= expected:
                    logger.info(f"[ROS] Received initial messages on all topics")
                    return
            time.sleep(0.05)

        with self._ros_lock:
            missing = expected - set(self._latest_ros_messages.keys())
        if missing:
            logger.warning(f"[ROS] Missing initial messages: {missing}")

    def close(self):
        """Cleanup all resources."""
        # Shutdown thread pool
        self.executor.shutdown(wait=False)

        # Unregister ROS subscribers
        for sub in self._ros_subscribers.values():
            try:
                sub.unregister()
            except Exception:
                pass
        self._ros_subscribers.clear()
        self._latest_ros_messages.clear()

        # Disconnect MQTT clients
        for cli in self._mqtt_clients.values():
            try:
                cli.loop_stop()
                cli.disconnect()
            except Exception:
                pass
        self._mqtt_clients.clear()

        # Clear image cache
        with self._image_cache_lock:
            self._image_cache.clear()

    def clear_history(self):
        """Clear message history."""
        with self._history_lock:
            self._history.clear()

    def get_history(self) -> List[Dict[str, Any]]:
        """Get recent message history."""
        with self._history_lock:
            return list(self._history)

    def process_all_topics(self) -> Dict[str, Any]:
        """Process all enabled topics - only if topic processing is enabled."""

        # Check if topic processing is enabled
        if not self.topic_processing_enabled:
            logger.debug("Topic processing is disabled, returning empty data")
            return {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "items": [],
                "tmp_dir": str(self.tmp_root),
                "topics_disabled": True
            }

        subs = self.topic_cfg.get("subscriptions", {})

        # Submit all topic reading tasks to thread pool
        futures = []

        # ROS topics
        for t in subs.get("ros_topics", []):
            if t.get("enabled", False):
                futures.append(self.executor.submit(self._read_ros_topic, t))

        # MQTT topics
        for t in subs.get("mqtt_topics", []):
            if t.get("enabled", False):
                futures.append(self.executor.submit(self._read_mqtt_topic, t))

        # File topics
        for t in subs.get("file_topics", []):
            if t.get("enabled", False):
                futures.append(self.executor.submit(self._read_file_topic, t))

        # Collect results as they complete
        out_items = []
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    if isinstance(result, list):
                        out_items.extend(result)
                    else:
                        out_items.append(result)
            except Exception as e:
                logger.error(f"Error processing topic: {e}")

        # Update history efficiently
        with self._history_lock:
            self._history.extend(out_items)

        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "items": out_items,
            "tmp_dir": str(self.tmp_root),
        }

    def _read_ros_topic(self, topic: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Routes to persistent cache or JIT reader based on config."""
        if self.persistent_subs_enabled:
            return self._read_ros_topic_persistent(topic)
        else:
            return self._read_ros_topic_jit(topic)

    def _read_ros_topic_persistent(self, topic: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Read latest cached ROS message from persistent subscriber."""
        if not self.ros_ok:
            return None

        name = topic.get("name")

        # Quick lock for reading
        with self._ros_lock:
            if name not in self._latest_ros_messages:
                return None
            msg, timestamp = self._latest_ros_messages[name]

        # Check message age (outside lock)
        age = time.time() - timestamp
        max_age = topic.get("max_message_age", 10.0)
        if max_age and age > float(max_age):
            return None

        try:
            msg_type = topic.get("message_type")

            # Time the conversion for performance monitoring
            convert_start = time.time()
            modality, data, extra = self._convert_ros_message(msg_type, msg)
            convert_time = (time.time() - convert_start) * 1000

            if convert_time > 100:  # Log if conversion takes > 100ms
                logger.warning(f"[PERF] Slow persistent conversion for {name}: {convert_time:.1f}ms")

            return {
                "id": f"ros::{name}::{int(timestamp*1000)}",
                "source": "ros",
                "name": name,
                "timestamp": datetime.fromtimestamp(timestamp).isoformat() + "Z",
                "modality": modality,
                "data": data,
                "analysis_hints": topic.get("analysis_hints", {}),
                "extra": extra,
                "message_age_seconds": round(age, 2),
            }
        except Exception as e:
            logger.error(f"[ROS-Persistent] Error converting {name}: {e}")
            return None

    def _read_ros_topic_jit(self, topic: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """On-demand (JIT) ROS topic read. Subscribes, waits, and unsubscribes."""
        if not self.ros_ok or rospy is None:
            return None

        name = topic.get("name")
        msg_type_str = topic.get("message_type")
        timeout = float(topic.get("timeout", 1.0))

        if not (name and msg_type_str):
            return None

        try:
            msg_cls = self._get_message_class(msg_type_str)

            start_time = time.time()
            msg = rospy.wait_for_message(name, msg_cls, timeout=timeout)
            timestamp = time.time()
            age = timestamp - start_time

            if msg is None:
                logger.debug(f"[ROS-JIT] Timeout waiting for message on {name}")
                return None

            convert_start = time.time()
            modality, data, extra = self._convert_ros_message(msg_type_str, msg)
            convert_time = (time.time() - convert_start) * 1000
            if convert_time > 100:
                logger.warning(f"[PERF] Slow JIT conversion for {name}: {convert_time:.1f}ms")

            return {
                "id": f"ros-jit::{name}::{int(timestamp*1000)}",
                "source": "ros",
                "name": name,
                "timestamp": datetime.fromtimestamp(timestamp).isoformat() + "Z",
                "modality": modality,
                "data": data,
                "analysis_hints": topic.get("analysis_hints", {}),
                "extra": extra,
                "message_age_seconds": round(age, 2),
            }

        except rospy.exceptions.ROSException as e:
            logger.warning(f"[ROS-JIT] Failed to get message from {name}: {e}")
            return None
        except Exception as e:
            logger.error(f"[ROS-JIT] Error processing {name}: {e}")
            return None

    def _convert_ros_message(self, msg_type: str, msg) -> Tuple[str, Any, Dict]:
        """Convert ROS message to (modality, data, extra) with optimized handlers."""
        extra = {"message_type": msg_type}

        # Fast path for common types
        if msg_type == "sensor_msgs/Image":
            return "image", self._save_ros_image_optimized(msg), extra

        if msg_type == "std_msgs/String":
            return "text", getattr(msg, "data", ""), extra

        if msg_type in ("std_msgs/Float64", "std_msgs/Int32", "std_msgs/Int64", "std_msgs/Bool"):
            return "text", str(getattr(msg, "data", "")), extra

        if msg_type == "turtlesim/Pose":
            return "structured", {
                "x": float(msg.x), "y": float(msg.y), "theta": float(msg.theta),
                "linear_velocity": float(msg.linear_velocity),
                "angular_velocity": float(msg.angular_velocity)
            }, extra

        if msg_type == "sensor_msgs/JointState":
            return "structured", {
                "name": list(msg.name), "position": list(msg.position),
                "velocity": list(msg.velocity), "effort": list(msg.effort)
            }, extra

        # Fallback to dict conversion
        try:
            return "structured", self._msg_to_dict(msg), extra
        except Exception:
            return "text", str(msg), extra

    def _save_ros_image_optimized(self, img_msg) -> str:
        """OPTIMIZED: Save ROS Image with resizing and compression."""
        save_start = time.time()

        try:
            # Convert using cv_bridge (fastest method)
            if self.bridge and cv2:
                cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

                # OPTIMIZATION 1: Resize if image is too large
                height, width = cv_img.shape[:2]
                if max(height, width) > self.MAX_IMAGE_DIMENSION:
                    scale = self.MAX_IMAGE_DIMENSION / max(height, width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)

                    logger.info(f"[IMG] Resizing {width}x{height} -> {new_width}x{new_height}")
                    cv_img = cv2.resize(cv_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

                # OPTIMIZATION 2: Use optimized JPEG encoding
                path = self._make_temp_path("ros_image", ".jpg")

                encode_params = [
                    cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY,
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1 if self.USE_FAST_JPEG else 0
                ]

                cv2.imwrite(path, cv_img, encode_params)

                # Log performance
                save_time = (time.time() - save_start) * 1000
                file_size = os.path.getsize(path) / 1024  # KB
                logger.info(f"[IMG] Saved in {save_time:.1f}ms, size: {file_size:.1f}KB")

                return path

            # Fallback to PIL
            elif Image:
                mode = "RGB" if "rgb" in img_msg.encoding.lower() else "L"
                img = Image.frombytes(mode, (img_msg.width, img_msg.height), bytes(img_msg.data))

                # OPTIMIZATION 1: Resize if needed
                if max(img.size) > self.MAX_IMAGE_DIMENSION:
                    img.thumbnail((self.MAX_IMAGE_DIMENSION, self.MAX_IMAGE_DIMENSION), Image.LANCZOS)
                    logger.info(f"[IMG] Resized to {img.size}")

                # OPTIMIZATION 2: Save as JPEG with compression
                path = self._make_temp_path("ros_image", ".jpg")
                img.save(path, "JPEG", quality=self.JPEG_QUALITY, optimize=True)

                save_time = (time.time() - save_start) * 1000
                file_size = os.path.getsize(path) / 1024
                logger.info(f"[IMG] Saved in {save_time:.1f}ms, size: {file_size:.1f}KB")

                return path
            else:
                raise RuntimeError("No image library available")

        except Exception as e:
            logger.error(f"[IMG] Error saving image: {e}")
            return self._save_ros_image_fallback(img_msg)

    def _save_ros_image_fallback(self, img_msg) -> str:
        """Fallback image saving method (original)."""
        if self.bridge and cv2:
            try:
                cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
                path = self._make_temp_path("ros_image", ".jpg")
                cv2.imwrite(path, cv_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                return path
            except Exception as e:
                logger.debug(f"cv_bridge fallback failed: {e}")

        if not Image:
            raise RuntimeError("PIL not available")

        mode = "RGB" if "rgb" in img_msg.encoding.lower() else "L"
        img = Image.frombytes(mode, (img_msg.width, img_msg.height), bytes(img_msg.data))
        path = self._make_temp_path("ros_image", ".png")
        img.save(path, "PNG", optimize=True)
        return path

    @lru_cache(maxsize=64)
    def _get_slots_cached(self, obj_type):
        """Cache object slots for repeated conversions."""
        return getattr(obj_type, "__slots__", [])

    def _msg_to_dict(self, msg) -> Dict:
        """Optimized recursive ROS message to dict conversion."""
        def convert(x):
            if hasattr(x, "__slots__"):
                slots = self._get_slots_cached(type(x))
                return {s: convert(getattr(x, s)) for s in slots}
            if isinstance(x, (list, tuple)):
                return [convert(v) for v in x]
            if isinstance(x, (bytes, bytearray)):
                return f"<{len(x)} bytes>"
            return x
        return convert(msg)

    def _read_mqtt_topic(self, topic: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Read MQTT topic with timeout."""
        if not mqtt:
            return None

        name = topic.get("name")
        host = topic.get("broker_host", "localhost")
        port = int(topic.get("broker_port", 1883))
        if not name:
            return None

        rx = queue.Queue(maxsize=1)
        client = self._get_mqtt_client(host, port, lambda c, u, m: rx.put_nowait((m.topic, m.payload, time.time())))

        if client is None:
            return None

        client.subscribe(name)
        timeout = float(topic.get("timeout", 1.0))

        try:
            _, payload, t = rx.get(timeout=timeout)
        except queue.Empty:
            return None
        finally:
            try:
                client.unsubscribe(name)
            except Exception:
                pass

        modality, data, extra = self._decode_mqtt_payload(payload)
        return {
            "id": f"mqtt::{name}::{int(t*1000)}",
            "source": "mqtt",
            "name": name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "modality": modality,
            "data": data,
            "analysis_hints": topic.get("analysis_hints", {}),
            "extra": extra,
        }

    def _get_mqtt_client(self, host: str, port: int, on_message_cb):
        """Get or create MQTT client with connection pooling."""
        key = (host, port)

        with self._mqtt_lock:
            if key in self._mqtt_clients:
                self._mqtt_clients[key].on_message = on_message_cb
                return self._mqtt_clients[key]

        try:
            cli = mqtt.Client()
            cli.on_message = on_message_cb
            cli.connect(host, port, keepalive=30)
            cli.loop_start()

            with self._mqtt_lock:
                self._mqtt_clients[key] = cli

            logger.info(f"[MQTT] Connected to {host}:{port}")
            return cli
        except Exception as e:
            logger.warning(f"[MQTT] Failed to connect to {host}:{port}: {e}")
            return None

    def _decode_mqtt_payload(self, payload: bytes) -> Tuple[str, Any, Dict]:
        """Optimized MQTT payload decoding."""
        # Try JSON first
        try:
            obj = json.loads(payload)

            # Check for embedded images
            for key in ("image_base64", "b64", "image"):
                b64 = obj.get(key)
                if isinstance(b64, str) and len(b64) > 100:
                    path = self._save_base64_image(b64)
                    return "image", path, {"format": "base64-in-json"}

            return "structured", obj, {"format": "json"}
        except:
            pass

        # Check for image
        if self._is_image_fast(payload):
            return "image", self._save_binary_image(payload), {"format": "binary"}

        # Fallback to text
        try:
            return "text", payload.decode("utf-8", errors="replace"), {"format": "text"}
        except:
            return "text", f"<{len(payload)} bytes>", {"format": "binary"}

    def _is_image_fast(self, data: bytes) -> bool:
        """Optimized image signature detection."""
        if len(data) < 12:
            return False

        if data.startswith(b"\x89PNG"):
            return True
        if data[:2] == b"\xff\xd8":
            return True
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return True
        if data[:2] in (b"BM", b"GI"):
            return True

        return False

    def _read_file_topic(self, topic: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Read file topic (log file or image directory)."""
        name = topic.get("name")
        pattern = topic.get("file_pattern")
        if not name:
            return None

        if pattern:
            # Image directory - get last 3 files
            matches = sorted(glob.glob(os.path.join(name, pattern)))[-3:]
            items = []
            for path in matches:
                try:
                    items.append({
                        "id": f"file::{path}::{int(os.path.getmtime(path)*1000)}",
                        "source": "file",
                        "name": path,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "modality": "image",
                        "data": self._copy_to_temp(path),
                        "analysis_hints": topic.get("analysis_hints", {}),
                    })
                except Exception as e:
                    logger.error(f"[FILE] Failed to copy {path}: {e}")
            return items if items else None
        else:
            # Text log file
            if not os.path.isfile(name):
                return None
            try:
                size = os.path.getsize(name)
                with open(name, "rb") as f:
                    if size > 4096:
                        f.seek(-4096, os.SEEK_END)
                    content = f.read().decode("utf-8", errors="replace")

                return [{
                    "id": f"file::{name}::{int(os.path.getmtime(name)*1000)}",
                    "source": "file",
                    "name": name,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "modality": "text",
                    "data": content,
                    "analysis_hints": topic.get("analysis_hints", {}),
                }]
            except Exception as e:
                logger.error(f"[FILE] Error reading {name}: {e}")
                return None

    def _save_base64_image(self, b64: str) -> str:
        """Optimized base64 image decoding and saving."""
        if "," in b64 and b64.strip().lower().startswith("data:"):
            b64 = b64.split(",", 1)[1]

        return self._save_binary_image(base64.b64decode(b64))

    def _save_binary_image(self, data: bytes) -> str:
        """Optimized binary image saving with format detection."""
        ext = ".jpg"
        if data.startswith(b"\x89PNG"):
            ext = ".png"
        elif data[:4] == b"RIFF" and len(data) > 12 and data[8:12] == b"WEBP":
            ext = ".webp"

        path = self._make_temp_path("image", ext)

        with open(path, "wb") as f:
            f.write(data)

        return path

    def _copy_to_temp(self, src: str) -> str:
        """Optimized file copy to temp directory."""
        ext = os.path.splitext(src)[1] or ".bin"
        dst = self._make_temp_path("file", ext)

        shutil.copy2(src, dst)
        return dst

    def _make_temp_path(self, prefix: str, ext: str) -> str:
        """Create temporary file path efficiently."""
        if not self.tmp_root.exists():
            self.tmp_root.mkdir(parents=True, exist_ok=True)

        fd, path = tempfile.mkstemp(prefix=f"{prefix}_", suffix=ext, dir=str(self.tmp_root))
        os.close(fd)
        return path

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass