#!/usr/bin/env bash
set -euo pipefail

PI_HOST="${1:?Usage: ./deploy.sh pi@<IP> <BT_MAC>}"
BT_MAC="${2:?Usage: ./deploy.sh pi@<IP> <BT_MAC>}"

echo "==> Syncing files to ${PI_HOST}:~/nova/"
rsync -avz --exclude '__pycache__' --exclude '.git' --exclude 'tests/' \
    ./ "${PI_HOST}:~/nova/"

echo "==> Installing system dependencies"
ssh "$PI_HOST" "sudo apt-get install -y --no-install-recommends \
    python3-pip libportaudio2 libasound2-dev bluetooth bluez"

echo "==> Installing Python dependencies"
ssh "$PI_HOST" "pip3 install --break-system-packages -r ~/nova/requirements_pi.txt"

echo "==> Downloading Piper voice model (if needed)"
ssh "$PI_HOST" "mkdir -p ~/nova/voices && \
    [ -f ~/nova/voices/en_US-lessac-medium.onnx ] || \
    wget -q --show-progress \
      -O ~/nova/voices/en_US-lessac-medium.onnx \
      'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx'"

echo "==> Setting SunFounder USB microphone as default PipeWire input"
ssh "$PI_HOST" "
    USB_SRC=\$(pactl list sources short 2>/dev/null | grep -i usb | awk '{print \$2}' | head -1)
    if [ -n \"\$USB_SRC\" ]; then
        pactl set-default-source \"\$USB_SRC\"
        echo \"Default input set to: \$USB_SRC\"
    else
        echo 'WARNING: USB microphone not detected. Plug it in and re-run deploy.sh.'
    fi
"

echo "==> Writing bt-speaker.service (MAC: ${BT_MAC})"
ssh "$PI_HOST" "sed 's/PLACEHOLDER_BT_MAC/${BT_MAC}/g' ~/nova/bt-speaker.service | \
    sudo tee /etc/systemd/system/bt-speaker.service > /dev/null"

echo "==> Installing nova.service"
ssh "$PI_HOST" "sudo cp ~/nova/nova.service /etc/systemd/system/nova.service"

echo "==> Enabling services"
ssh "$PI_HOST" "sudo systemctl daemon-reload && sudo systemctl enable bt-speaker nova"

echo ""
echo "==> Deploy complete."
echo ""
echo "If this is your first deploy, pair the Bluetooth speaker first:"
echo "  ssh ${PI_HOST}"
echo "  bluetoothctl"
echo "    power on"
echo "    agent on"
echo "    scan on          # wait for speaker MAC to appear"
echo "    pair   ${BT_MAC}"
echo "    trust  ${BT_MAC}"
echo "    connect ${BT_MAC}"
echo "    exit"
echo ""
read -rp "Start Nova now? [y/N] " answer
if [[ "${answer,,}" == "y" ]]; then
    ssh "$PI_HOST" "sudo systemctl start bt-speaker && sleep 2 && sudo systemctl start nova"
    echo "Nova started. Follow logs: ssh ${PI_HOST} 'journalctl -u nova -f'"
fi
