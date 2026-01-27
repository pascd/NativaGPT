from NativaGPT.lib.config_manager import ConfigManager
from NativaGPT.scripts.nativa import NativaGPT

from NativaGPT.lib.coloring_logger import logger

import subprocess, sys
import signal
import time
import os
import json
import requests
import socket

CONFIG_MANAGER_FILE="/home/pedrodias/Documents/git-repos/NativaGPT/config/config_default.json"

class EnhancedNativaGPTStarter:
    """Enhanced starter class with VLM integration and better management"""

    def __init__(self):
        self.nativa = None
        self.launch_process = None
        self.service_processes = []
        self.vlm_services = []

    def launch_dependencies(self):
        """Launch all necessary API dependencies including VLM services"""
        logger.info("Launching enhanced dependencies with VLM support...")

        # First, try the original bash script
        if self._try_bash_script():
            return True

        # If bash script fails, try to launch services individually
        logger.info("Bash script failed, attempting to launch services individually...")
        return self._launch_services_individually()

    def _try_bash_script(self):
        """Try to run the original bash script with environment fixes"""
        command = "/home/pedrodias/Documents/git-repos/NativaGPT/NativaGPT/bash/launch_all_api.sh"

        if not os.path.exists(command):
            logger.error(f"Launch script not found: {command}")
            return False

        try:
            # Set up environment for terminal applications
            env = os.environ.copy()
            env['DISPLAY'] = ':0'
            env['TERM'] = 'xterm'

            logger.info("Trying to run bash script with fixed environment...")

            # Try with nohup to detach from terminal
            nohup_command = f"nohup {command} > /tmp/nativa_launch.log 2>&1 &"
            result = subprocess.run(
                nohup_command,
                shell=True,
                executable='/bin/bash',
                env=env,
                timeout=15
            )

            if result.returncode == 0:
                logger.info("Bash script launched successfully")
                time.sleep(8)  # Give more time for VLM services
                return self._verify_services_running()

            # Try with screen if available
            if self._try_screen_launch(command):
                return True

            return False

        except subprocess.TimeoutExpired:
            logger.info("Bash script is running in background (timeout reached)")
            return self._verify_services_running()
        except Exception as e:
            logger.error(f"Error running bash script: {e}")
            return False

    def _try_screen_launch(self, command):
        """Try launching with screen if available"""
        try:
            screen_check = subprocess.run(['which', 'screen'],
                                        capture_output=True,
                                        text=True)

            if screen_check.returncode == 0:
                logger.info("Trying to launch with screen...")
                screen_command = f"screen -dmS nativa_services {command}"
                result = subprocess.run(screen_command, shell=True, timeout=5)

                if result.returncode == 0:
                    logger.info("Services launched in screen session 'nativa_services'")
                    time.sleep(8)
                    return self._verify_services_running()

            return False

        except Exception as e:
            logger.error(f"Error with screen launch: {e}")
            return False

    def _launch_services_individually(self):
        """Launch services individually including VLM services"""
        logger.info("Launching enhanced services individually...")

        # Define services including VLM-related ones
        services = [
            {
                'name': 'Ollama LLM API',
                'command': 'ollama serve',
                'port': 11434,
                'cwd': None,
                'required': True
            },
            {
                'name': 'STT API',
                'command': 'python -m whisper_server --port 8030',
                'port': 8030,
                'cwd': None,
                'required': False
            },
            {
                'name': 'TTS API',
                'command': 'python -m xtts_server --port 8020',
                'port': 8020,
                'cwd': None,
                'required': False
            }
        ]

        success_count = 0
        required_count = 0

        for service in services:
            if service.get('required', False):
                required_count += 1

            if self._launch_single_service(service):
                success_count += 1
            else:
                if service.get('required', False):
                    logger.error(f"Required service failed to launch: {service['name']}")
                else:
                    logger.warning(f"Optional service failed to launch: {service['name']}")

        # Check if we have minimum required services
        if success_count >= required_count:
            logger.info(f"Successfully launched {success_count}/{len(services)} services")
            time.sleep(8)  # Give services time to initialize
            return True
        else:
            logger.error(f"Failed to launch required services. Got {success_count}, needed {required_count}")
            return False

    def _launch_single_service(self, service):
        """Launch a single service in the background"""
        try:
            logger.info(f"Starting {service['name']}...")

            # Check if service is already running
            if self._check_port(service['port']):
                logger.info(f"✓ {service['name']} is already running on port {service['port']}")
                return True

            # Launch the service
            process = subprocess.Popen(
                service['command'],
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=service['cwd'],
                start_new_session=True
            )

            self.service_processes.append({
                'name': service['name'],
                'process': process,
                'port': service['port']
            })

            # Give it a moment to start
            time.sleep(3)

            # Check if it's running
            if process.poll() is None:
                logger.info(f"✓ {service['name']} process started (PID: {process.pid})")
                return True
            else:
                logger.error(f"✗ {service['name']} process exited immediately")
                return False

        except Exception as e:
            logger.error(f"Error launching {service['name']}: {e}")
            return False

    def _check_port(self, port):
        """Check if a port is responsive (enhanced version)"""
        try:
            # First, try a lightweight HTTP GET to verify responsiveness
            response = requests.get(f"http://localhost:{port}", timeout=2)
            return response.status_code in [200, 404]
        except:
            pass

        # Fallback: raw socket check (basic port open test)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result == 0
        except:
            return False

    def _verify_services_running(self):
        """Verify that services are actually running"""
        services_to_check = [
            (11434, "Ollama LLM API", True),  # Required
            (8030, "STT API", False),         # Optional
            (8020, "TTS API", False)          # Optional
        ]

        running_services = 0
        required_running = 0
        total_required = 0

        logger.info("Verifying enhanced services are running...")
        for port, name, required in services_to_check:
            if required:
                total_required += 1

            if self._check_port(port):
                logger.info(f"✓ {name} is running on port {port}")
                running_services += 1
                if required:
                    required_running += 1
            else:
                if required:
                    logger.error(f"✗ {name} (REQUIRED) is not responding on port {port}")
                else:
                    logger.warning(f"✗ {name} (optional) is not responding on port {port}")

        # We need at least the required services
        if required_running >= total_required:
            logger.info(f"Minimum required services running: {required_running}/{total_required}")
            logger.info(f"Total services running: {running_services}/{len(services_to_check)}")
            return True
        else:
            logger.error(f"Not enough required services running: {required_running}/{total_required}")
            return False

    def initialize_nativa(self):
        """Initialize Enhanced NativaGPT with VLM support"""
        try:
            # Initialize config and handlers
            config_manager = ConfigManager(CONFIG_MANAGER_FILE)
            logger.info(f"Loading enhanced config file: {CONFIG_MANAGER_FILE}")
            config = config_manager.get()

            # Validate enhanced configuration
            if not self._validate_enhanced_config(config):
                logger.error("Enhanced configuration validation failed")
                return False

            # Create VLM functions file if it doesn't exist
            self._ensure_vlm_functions_file(config)

            # Initialize Enhanced NativaGPT
            logger.info("Initializing Enhanced NativaGPT with VLM integration...")
            self.nativa = NativaGPT(config)

            logger.info("Enhanced NativaGPT initialization complete")
            return True

        except Exception as e:
            logger.error(f"Error initializing Enhanced NativaGPT: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def _validate_enhanced_config(self, config):
        """Enhanced configuration validation including VLM settings"""
        required_sections = ['nativa_gpt', 'llm_config', 'vlm_config', 'topic_config']

        for section in required_sections:
            if section not in config:
                logger.warning(f"Missing configuration section: {section}")

        # Check VLM specific configuration
        vlm_config = config.get('vlm_config', {})
        if vlm_config.get('enabled', True):
            output_dir = vlm_config.get('output_directory', '/tmp/nativa_vlm_output')
            try:
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"VLM output directory ready: {output_dir}")
            except Exception as e:
                logger.error(f"Cannot create VLM output directory: {e}")
                return False

        # Check if vision model is configured
        vision_model = config.get('llm_config', {}).get('vision_model')
        if not vision_model:
            logger.warning("No vision model configured - VLM features may not work")

        return True

    def _ensure_vlm_functions_file(self, config):
        """Ensure VLM functions file exists"""
        try:
            vlm_functions_path = "/home/pedrodias/Documents/git-repos/NativaGPT/config/functions/vlm_functions.json"

            if not os.path.exists(vlm_functions_path):
                logger.info("Creating VLM functions file...")

                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(vlm_functions_path), exist_ok=True)

                # Create basic VLM functions file
                vlm_functions = [
                    {
                        "function": {
                            "name": "Analyze Image",
                            "description": "Analyze an image using Vision Language Model",
                            "execution": "vlm",
                            "command": "vlm_handler.analyze_image",
                            "location": ""
                        }
                    }
                ]

                with open(vlm_functions_path, 'w') as f:
                    json.dump(vlm_functions, f, indent=2)

                logger.info(f"Created VLM functions file: {vlm_functions_path}")

        except Exception as e:
            logger.error(f"Error ensuring VLM functions file: {e}")

    def start(self):
        """Start the enhanced system with VLM support"""
        try:
            logger.info("=" * 70)
            logger.info("Starting Enhanced NativaGPT System with Advanced VLM Integration")
            logger.info("=" * 70)

            # Launch dependencies
            logger.info("Step 1: Launching enhanced dependencies...")
            if not self.launch_dependencies():
                logger.error("Failed to launch dependencies")

                # Ask user if they want to continue anyway
                try:
                    response = input("Dependencies failed to launch. Continue anyway? (y/N): ").strip().lower()
                    if response != 'y':
                        logger.info("Exiting due to dependency launch failure")
                        return False
                    else:
                        logger.warning("Continuing without full dependency verification...")
                except (EOFError, KeyboardInterrupt):
                    logger.info("Exiting...")
                    return False

            # Initialize Enhanced NativaGPT
            logger.info("Step 2: Initializing Enhanced NativaGPT with VLM...")
            if not self.initialize_nativa():
                logger.error("Failed to initialize Enhanced NativaGPT, exiting...")
                return False

            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            # Show system status
            self._show_system_status()

            # Start the main system
            logger.info("Step 3: Starting enhanced main system...")
            logger.info("=" * 70)

            # Start NativaGPT
            self.nativa.start()

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
            self._shutdown()
        except Exception as e:
            logger.error(f"Error starting enhanced system: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self._shutdown()
            return False

    def run_interactive_mode(self):
        """Run enhanced interactive mode with VLM capabilities"""
        if not self.nativa:
            logger.error("Enhanced NativaGPT not initialized")
            return False

        self._show_enhanced_welcome()

        try:
            self.nativa.start()  # Full interactive loop runs here
        except KeyboardInterrupt:
            logger.info("User interrupted interactive mode")
            self._shutdown()

    def _show_enhanced_welcome(self):
        """Show enhanced welcome message with VLM features"""
        print("\n" + "🤖" * 20)
        print("🎉 Enhanced NativaGPT with Advanced VLM Integration 🎉")
        print("🤖" * 20)
        print()
        print("🔥 New VLM Features Available:")
        print("   📸 Smart camera analysis with shortcuts")
        print("   🔍 Advanced object detection")
        print("   🎯 Scene understanding and description")
        print("   📊 Image comparison and change detection")
        print("   🎨 Image generation capabilities")
        print("   🤖 Intelligent shortcuts and auto-analysis")
        print()
        print("💡 Try these enhanced commands:")
        print("   • 'camera' - Quick camera analysis")
        print("   • 'find water bottle in image' - Object detection")
        print("   • 'what do you see' - Scene description")
        print("   • 'what changed' - Compare with previous image")
        print("   • 'vlm history' - Show analysis history")
        print()
        print("Type 'help' for complete command list!")
        print("=" * 50)

    def _show_system_status(self):
        """Show enhanced system status"""
        logger.info("Enhanced System Status:")
        logger.info("-" * 30)

        # Check core services
        core_services = [
            (11434, "Ollama LLM/VLM"),
            (8030, "Speech-to-Text"),
            (8020, "Text-to-Speech")
        ]

        for port, name in core_services:
            status = "✅ Running" if self._check_port(port) else "❌ Not Running"
            logger.info(f"{name:20} : {status}")

        # Check VLM capabilities
        if hasattr(self.nativa, 'vlm_handler'):
            logger.info(f"{'VLM Handler':20} : ✅ Initialized")
        else:
            logger.info(f"{'VLM Handler':20} : ❌ Not Available")

        # Check camera topics
        if hasattr(self.nativa, 'smart_shortcuts'):
            camera_count = len(self.nativa.smart_shortcuts.get('camera_topics', []))
            logger.info(f"{'Camera Topics':20} : {camera_count} detected")

        logger.info("-" * 30)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown()

    def _shutdown(self):
        """Enhanced graceful shutdown with VLM cleanup"""
        logger.info("Shutting down Enhanced NativaGPT system...")

        try:
            # Stop NativaGPT first (includes VLM handler cleanup)
            if self.nativa:
                logger.info("Stopping Enhanced NativaGPT...")
                try:
                    if hasattr(self.nativa, 'stop'):
                        self.nativa.stop()
                    if hasattr(self.nativa, 'cleanup'):
                        self.nativa.cleanup()
                except Exception as e:
                    logger.error(f"Error stopping Enhanced NativaGPT: {e}")

            # Stop individual service processes
            for service_info in self.service_processes:
                try:
                    process = service_info['process']
                    name = service_info['name']

                    if process.poll() is None:
                        logger.info(f"Stopping {name}...")
                        process.terminate()

                        # Wait for graceful shutdown
                        try:
                            process.wait(timeout=10)  # Longer timeout for VLM services
                            logger.info(f"✓ {name} stopped gracefully")
                        except subprocess.TimeoutExpired:
                            logger.info(f"Force killing {name}...")
                            process.kill()
                            process.wait()

                except Exception as e:
                    logger.error(f"Error stopping service {service_info['name']}: {e}")

            # Stop launch process if it exists
            if self.launch_process and self.launch_process.poll() is None:
                logger.info("Stopping launch process...")
                try:
                    os.killpg(os.getpgid(self.launch_process.pid), signal.SIGTERM)
                    time.sleep(3)
                    if self.launch_process.poll() is None:
                        os.killpg(os.getpgid(self.launch_process.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass

            logger.info("Exiting process now.")
            sys.exit(0)

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def show_running_services(self):
        """Show which enhanced services are currently running"""
        logger.info("Checking enhanced service status...")

        services_to_check = [
            (11434, "Ollama LLM/VLM API", True),
            (8030, "STT API", False),
            (8020, "TTS API", False)
        ]

        for port, name, required in services_to_check:
            status_icon = "✓" if self._check_port(port) else "✗"
            required_text = "(REQUIRED)" if required else "(optional)"
            status_text = "running" if self._check_port(port) else "not running"

            logger.info(f"{status_icon} {name} {required_text} is {status_text} on port {port}")


if __name__ == '__main__':
    starter = EnhancedNativaGPTStarter()

    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            # Test mode
            logger.info("Running enhanced test mode...")
            if starter.launch_dependencies() and starter.initialize_nativa():
                starter.show_running_services()

                logger.info("Enhanced test mode active. Press Ctrl+C to exit.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass

            starter._shutdown()

        elif sys.argv[1] == "--interactive":
            # Enhanced interactive mode
            logger.info("Starting enhanced interactive mode with VLM...")
            if starter.launch_dependencies() and starter.initialize_nativa():
                starter.run_interactive_mode()
            starter._shutdown()

        elif sys.argv[1] == "--no-deps":
            # Skip dependency launch for testing
            logger.info("Starting enhanced mode without dependencies...")
            if starter.initialize_nativa():
                starter.run_interactive_mode()
            starter._shutdown()

        elif sys.argv[1] == "--deps-only":
            # Only launch dependencies for testing
            logger.info("Launching enhanced dependencies only...")
            if starter.launch_dependencies():
                starter.show_running_services()
                logger.info("Enhanced dependencies launched. Press Ctrl+C to stop.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
            starter._shutdown()

        elif sys.argv[1] == "--status":
            # Show service status only
            starter.show_running_services()

        elif sys.argv[1] == "--vlm-test":
            # VLM-specific test mode
            logger.info("Running VLM integration test...")
            if starter.launch_dependencies() and starter.initialize_nativa():
                # Test VLM capabilities
                if hasattr(starter.nativa, 'vlm_handler'):
                    logger.info("✅ VLM Handler available")
                    stats = starter.nativa.vlm_handler.get_stats()
                    logger.info(f"VLM Stats: {stats}")
                else:
                    logger.error("❌ VLM Handler not available")

                # Test camera topics detection
                if hasattr(starter.nativa, 'smart_shortcuts'):
                    camera_topics = starter.nativa.smart_shortcuts.get('camera_topics', [])
                    logger.info(f"Detected camera topics: {camera_topics}")

            starter._shutdown()

        elif sys.argv[1] == "--help":
            print("Enhanced NativaGPT Launcher with VLM Integration")
            print("Usage:")
            print("  python start_nativa.py                 - Start full system (background mode)")
            print("  python start_nativa.py --interactive   - Enhanced interactive mode with VLM")
            print("  python start_nativa.py --test          - Test mode")
            print("  python start_nativa.py --vlm-test      - VLM integration test")
            print("  python start_nativa.py --no-deps       - Interactive mode without dependencies")
            print("  python start_nativa.py --deps-only     - Only launch dependencies")
            print("  python start_nativa.py --status        - Show service status")
            print("  python start_nativa.py --help          - Show this help")
            print()
            print("Enhanced Features:")
            print("  🔥 Advanced VLM integration with vision analysis")
            print("  📸 Smart camera topic detection and shortcuts")
            print("  🔍 Object detection and scene understanding")
            print("  📊 Image comparison and change detection")
            print("  🎨 Image generation capabilities")
            print("  🤖 Intelligent shortcuts and auto-analysis")
        else:
            logger.error(f"Unknown argument: {sys.argv[1]}")
            sys.exit(1)
    else:
        # Normal startup (background mode)
        starter.start()