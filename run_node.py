# run_node.py
"""
Main entry point for running ScrapeHive nodes
"""

import os
import sys
import click
import logging
from typing import List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nodes.ai_node import AINode
from nodes.scraping_node import ScrapingNode
from nodes.scheduler_node import SchedulerNode
from nodes.view_master import ViewMasterNode

@click.command()
@click.option('--type', '-t', 
              type=click.Choice(['ai', 'scraping', 'scheduler', 'view-master']),
              required=True,
              help='Type of node to run')
@click.option('--id', '-i', 'node_id',
              help='Node ID (auto-generated if not provided)')
@click.option('--master-ip', '-m',
              default='localhost',
              help='Master/Redis IP address')
@click.option('--capabilities', '-c',
              multiple=True,
              default=['reddit', 'twitter'],
              help='Scraping capabilities (for scraping nodes)')
@click.option('--log-level', '-l',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO',
              help='Logging level')
def main(type: str, node_id: str, master_ip: str, capabilities: List[str], log_level: str):
    """
    ScrapeHive - Distributed Scraping and AI Processing System
    
    Examples:
    
    \b
    # Run AI node on GPU machine
    python run_node.py --type ai --id ai-gpu-01
    
    \b
    # Run scraping node with specific capabilities
    python run_node.py --type scraping --capabilities reddit twitter instagram
    
    \b
    # Run scheduler node
    python run_node.py --type scheduler
    
    \b
    # Run view master (dashboard)
    python run_node.py --type view-master
    """
    
    # Set log level
    os.environ['LOG_LEVEL'] = log_level
    
    # Auto-generate node ID if not provided
    if not node_id:
        import socket
        hostname = socket.gethostname()
        node_id = f"{type}-{hostname}"
    
    # Create and run appropriate node
    try:
        if type == 'ai':
            node = AINode(node_id, master_ip)
        elif type == 'scraping':
            node = ScrapingNode(node_id, master_ip, list(capabilities))
        elif type == 'scheduler':
            node = SchedulerNode(node_id, master_ip)
        elif type == 'view-master':
            node = ViewMasterNode(node_id, master_ip)
        
        # Run the node
        node.run()
        
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        logging.error(f"Failed to start node: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

# scripts/setup.sh
#!/bin/bash
# ScrapeHive Setup Script

echo "🕷️ ScrapeHive Setup"
echo "=================="

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install base requirements
echo "Installing base requirements..."
pip install -r requirements/base.txt

# Ask which node type to setup
echo ""
echo "Which node type are you setting up?"
echo "1) AI Node (GPU)"
echo "2) Scraping Node"
echo "3) Scheduler Node"
echo "4) View Master (Dashboard)"
echo "5) All (Development)"
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo "Installing AI node requirements..."
        pip install -r requirements/ai.txt
        ;;
    2)
        echo "Installing scraping node requirements..."
        pip install -r requirements/scraping.txt
        ;;
    3)
        echo "Base requirements are sufficient for scheduler node"
        ;;
    4)
        echo "Installing view master requirements..."
        pip install -r requirements/master.txt
        ;;
    5)
        echo "Installing all requirements..."
        pip install -r requirements/ai.txt
        pip install -r requirements/scraping.txt
        pip install -r requirements/master.txt
        ;;
esac

# Create directories
echo "Creating directories..."
mkdir -p logs
mkdir -p data/scraped
mkdir -p data/processed
mkdir -p models/weights

# Copy environment file
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "Please edit .env with your configuration"
fi

# Check Redis
echo "Checking Redis..."
if command -v redis-cli &> /dev/null; then
    echo "✓ Redis is installed"
else
    echo "✗ Redis is not installed"
    echo "Please install Redis:"
    echo "  Ubuntu/Debian: sudo apt-get install redis-server"
    echo "  macOS: brew install redis"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run a node:"
echo "  source venv/bin/activate"
echo "  python run_node.py --type <node-type>"
echo ""
echo "For help:"
echo "  python run_node.py --help"

# scripts/deploy_node.py
#!/usr/bin/env python3
"""
Deploy script for easy node setup on new machines
"""

import os
import sys
import subprocess
import platform
import argparse

def check_python():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8+ required")
        sys.exit(1)
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")

def install_redis():
    """Install Redis if not present"""
    try:
        subprocess.run(['redis-cli', 'ping'], capture_output=True, check=True)
        print("✓ Redis already installed")
    except:
        print("Installing Redis...")
        system = platform.system()
        
        if system == 'Linux':
            subprocess.run(['sudo', 'apt-get', 'update'])
            subprocess.run(['sudo', 'apt-get', 'install', '-y', 'redis-server'])
        elif system == 'Darwin':  # macOS
            subprocess.run(['brew', 'install', 'redis'])
        else:
            print("Please install Redis manually")

def setup_node(node_type, master_ip):
    """Setup specific node type"""
    print(f"\nSetting up {node_type} node...")
    
    # Clone repo
    if not os.path.exists('scrapehive'):
        print("Cloning repository...")
        subprocess.run(['git', 'clone', 'https://github.com/yourusername/scrapehive.git'])
    
    os.chdir('scrapehive')
    
    # Run setup
    subprocess.run(['bash', 'scripts/setup.sh'])
    
    # Create systemd service (Linux only)
    if platform.system() == 'Linux':
        create_service(node_type)

def create_service(node_type):
    """Create systemd service"""
    service_content = f"""[Unit]
Description=ScrapeHive {node_type} Node
After=network.target

[Service]
Type=simple
User={os.getenv('USER')}
WorkingDirectory={os.getcwd()}
ExecStart={os.path.join(os.getcwd(), 'venv/bin/python')} run_node.py --type {node_type}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    service_file = f'/tmp/scrapehive-{node_type}.service'
    with open(service_file, 'w') as f:
        f.write(service_content)
    
    print(f"Service file created at {service_file}")
    print("To install:")
    print(f"  sudo cp {service_file} /etc/systemd/system/")
    print(f"  sudo systemctl enable scrapehive-{node_type}")
    print(f"  sudo systemctl start scrapehive-{node_type}")

def main():
    parser = argparse.ArgumentParser(description='Deploy ScrapeHive node')
    parser.add_argument('--type', required=True, 
                       choices=['ai', 'scraping', 'scheduler', 'view-master'])
    parser.add_argument('--master-ip', default='localhost')
    args = parser.parse_args()
    
    check_python()
    install_redis()
    setup_node(args.type, args.master_ip)
    
    print("\n✅ Deployment complete!")
    print(f"\nTo run: python run_node.py --type {args.type} --master-ip {args.master_ip}")

if __name__ == '__main__':
    main()