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

echo "==> Installing ReSpeaker 2-Mics HAT driver (HinTak fork for Pi 5 kernel 6.6)"
ssh "$PI_HOST" "
    if dpkg -l | grep -q seeed-voicecard 2>/dev/null; then
        echo 'ReSpeaker driver already installed, skipping.'
    else
        sudo apt-get install -y --no-install-recommends git raspberrypi-kernel-headers dkms
        if [ ! -d ~/seeed-voicecard ]; then
            git clone https://github.com/HinTak/seeed-voicecard ~/seeed-voicecard
        fi
        cd ~/seeed-voicecard && git checkout v6.6
        sudo ./install.sh
        echo 'ReSpeaker driver installed. Reboot required before audio works.'
    fi
"

echo "==> Enabling I2C (required by ReSpeaker WM8960 codec)"
ssh "$PI_HOST" "sudo raspi-config nonint do_i2c 0"

echo "==> Configuring ReSpeaker as default audio input"
ssh "$PI_HOST" "
    SEEED_SRC=\$(pactl list sources short 2>/dev/null | grep -i seeed | awk '{print \$2}' | head -1)
    if [ -n \"\$SEEED_SRC\" ]; then
        pactl set-default-source \"\$SEEED_SRC\"
        echo \"Default input set to: \$SEEED_SRC\"
    else
        echo 'NOTE: ReSpeaker not yet visible to PipeWire — a reboot is needed first.'
        echo 'After reboot, re-run deploy.sh to complete audio configuration.'
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
read -rp "Reboot Pi now to activate ReSpeaker driver? [y/N] " reboot_answer
if [[ "${reboot_answer,,}" == "y" ]]; then
    ssh "$PI_HOST" "sudo reboot" || true
    echo ""
    echo "Pi is rebooting. Wait ~60 seconds, then re-run deploy.sh to finalize audio config:"
    echo "  ./deploy.sh ${PI_HOST} ${BT_MAC}"
    exit 0
fi

echo ""
read -rp "Start Nova now? [y/N] " answer
if [[ "${answer,,}" == "y" ]]; then
    ssh "$PI_HOST" "sudo systemctl start bt-speaker && sleep 2 && sudo systemctl start nova"
    echo "Nova started. Follow logs: ssh ${PI_HOST} 'journalctl -u nova -f'"
fi
