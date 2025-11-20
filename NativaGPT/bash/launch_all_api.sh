#!/bin/bash

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "${SCRIPT_DIR}"

# Function to detect the available terminal emulator
get_terminal() {
    if command -v gnome-terminal &> /dev/null; then
        echo "gnome-terminal"
    elif command -v xterm &> /dev/null; then
        echo "xterm"
    elif command -v konsole &> /dev/null; then
        echo "konsole"
    else
        echo "none"
    fi
}

# Get the available terminal
TERMINAL=$(get_terminal)

if [ "$TERMINAL" = "none" ]; then
    echo "No supported terminal emulator found!"
    exit 1
fi

echo "Using terminal: $TERMINAL"
echo "Starting services..."

# Function to launch a script in a new terminal
launch_script() {
    local title=$1
    local script_path=$2

    case $TERMINAL in
        "gnome-terminal")
            gnome-terminal --title="$title" -- bash -c "$script_path; exec bash"
            ;;
        "xterm")
            xterm -T "$title" -e "bash -c '$script_path; exec bash'" &
            ;;
        "konsole")
            konsole --new-tab --title "$title" -e bash -c "$script_path; exec bash" &
            ;;
    esac

    # Wait a bit before launching next terminal
    sleep 2
}

# Launch each service bash script
echo "Starting STT restAPI..."
launch_script "Speech-to-Text restAPI" "${SCRIPT_DIR}/launch_stt_restAPI.sh"

#echo "Starting Kobold..."
#launch_script "Kobold" "${SCRIPT_DIR}/launch_kobold_restAPI.sh"

echo "Starting XTTS restAPI..."
launch_script "XTTS-API-SERVER" "${SCRIPT_DIR}/launch_tts_xtts.sh"

# echo "Starting LMSTUDIO restAPI..."
# launch_script "LMStudio CLI" "${SCRIPT_DIR}/launch_llm_lms.sh"

echo "All services started!"