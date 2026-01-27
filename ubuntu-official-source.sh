#!/bin/bash
# Ubuntu 22.04 (Jammy Jellyfish) official source configuration script
# Run this script with root privileges (sudo)

# Define color output (optional, for better readability)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color (reset to default)

# Check if the script is run as root
if [ "$(id -u)" -ne 0 ]; then
    echo -e "${RED}Error: Please run this script with root privileges (sudo ./script_name.sh)${NC}"
    exit 1
fi

# Backup the original sources.list file (prevent irreversible changes)
BACKUP_FILE="/etc/apt/sources.list.bak.$(date +%Y%m%d%H%M%S)"
cp /etc/apt/sources.list "$BACKUP_FILE"
echo -e "${YELLOW}Original sources file backed up to: $BACKUP_FILE${NC}"

# Write Ubuntu Jammy official sources (global source by default, can replace with domestic mirrors like Tsinghua/Aliyun)
cat > /etc/apt/sources.list << EOF
# Ubuntu 22.04 (Jammy Jellyfish) Official Sources
deb http://archive.ubuntu.com/ubuntu/ jammy main restricted universe multiverse
deb http://archive.ubuntu.com/ubuntu/ jammy-updates main restricted universe multiverse
deb http://archive.ubuntu.com/ubuntu/ jammy-backports main restricted universe multiverse
deb http://security.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse

# Comment out source code repositories (remove the # to enable if needed)
# deb-src http://archive.ubuntu.com/ubuntu/ jammy main restricted universe multiverse
# deb-src http://archive.ubuntu.com/ubuntu/ jammy-updates main restricted universe multiverse
# deb-src http://archive.ubuntu.com/ubuntu/ jammy-backports main restricted universe multiverse
# deb-src http://security.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse
EOF

# Update apt cache
echo -e "${GREEN}Updating apt cache...${NC}"
apt update -y

# Check if the update was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Ubuntu Jammy source configuration completed, and apt cache updated successfully!${NC}"
else
    echo -e "${RED}Source configuration completed, but apt cache update failed. Please check network or source address!${NC}"
fi
