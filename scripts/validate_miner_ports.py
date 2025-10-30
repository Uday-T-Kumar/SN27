#!/usr/bin/env python3
"""
Port validation script for Subnet 27 miner
Tests if multiple ports are accessible from external internet
"""

import socket
import subprocess
import sys
import time
import threading
import requests
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Optional, Tuple


class SimpleTestServer:
    """Simple HTTP server for testing port accessibility"""

    def __init__(self, port: int):
        self.port = port
        self.server = None
        self.thread = None
        self.test_id = f"test_{port}_{int(time.time())}"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                'status': 'ok',
                'port': self.server.server_port,
                'test_id': self.server.test_id
            }
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, format, *args):
            pass  # Silence logs

    def start(self) -> bool:
        """Start the test server"""
        try:
            self.server = HTTPServer(('0.0.0.0', self.port), self.Handler)
            self.server.test_id = self.test_id
            self.thread = threading.Thread(target=self.server.serve_forever)
            self.thread.daemon = True
            self.thread.start()
            return True
        except Exception as e:
            print(f"❌ Cannot start server on port {self.port}: {e}")
            return False

    def stop(self):
        """Stop the test server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()


class PortValidator:
    """Simple port validator using external checking service"""

    def __init__(self, ssh_port=4444, axon_port=8091, external_ports=None):
        if external_ports is None:
            external_ports = [27015, 27016, 27017, 27018]

        self.ports = {
            f'SSH ({ssh_port})': ssh_port,
            f'Axon ({axon_port})': axon_port,
        }

        # Add multiple external ports
        for port in external_ports:
            self.ports[f'External ({port})'] = port

        self.ssh_port = ssh_port
        self.axon_port = axon_port
        self.external_ports = external_ports
        self.servers = {}
        self.public_ip = None

    def get_public_ip(self) -> Optional[str]:
        """Get public IP address"""
        services = [
            'https://api.ipify.org',
            'https://ifconfig.me/ip',
            'https://checkip.amazonaws.com'
        ]

        for service in services:
            try:
                response = requests.get(service, timeout=5)
                if response.status_code == 200:
                    ip = response.text.strip()
                    # Validate IP format
                    socket.inet_aton(ip)
                    return ip
            except Exception as e:
                logging.warning(f"Failed to get public IP from {service}: {e}")
                continue
        return None

    def check_port_external(self, port: int, timeout: int = 10) -> Tuple[bool, str]:
        """
        Check if port is accessible from external internet using online service
        """
        if not self.public_ip:
            return False, "No public IP available"

        # Method 1: Use portchecker.io API (no auth required)
        try:
            url = f"https://ports.yougetsignal.com/check-port.php"
            data = {
                'remoteAddress': self.public_ip,
                'portNumber': port
            }
            headers = {
                'User-Agent': 'Mozilla/5.0',
                'Accept': 'application/json'
            }

            response = requests.post(url, data=data, headers=headers, timeout=timeout)
            if response.status_code == 200:
                result = response.json()
                if result.get('open'):
                    return True, "Port is open"
                else:
                    return False, "Port appears closed from internet"
        except Exception:
            pass

        # Method 2: Try direct connection test
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((self.public_ip, port))
            sock.close()

            if result == 0:
                return True, "Direct connection successful"
            else:
                return False, "Connection refused or timeout"
        except Exception as e:
            return False, f"Connection test failed: {str(e)}"

    def check_firewall(self) -> Dict:
        """Check basic firewall status"""
        results = {}

        # Check UFW
        try:
            result = subprocess.run(['sudo', 'ufw', 'status'],
                                  capture_output=True, text=True, timeout=5)
            if 'Status: active' in result.stdout:
                results['firewall'] = 'ufw'
                results['active'] = True
                results['rules'] = []

                for name, port in self.ports.items():
                    if str(port) in result.stdout:
                        results['rules'].append(port)
            else:
                results['firewall'] = 'ufw'
                results['active'] = False
        except Exception:
            results['firewall'] = 'unknown'
            results['active'] = False

        return results

    def start_test_servers(self) -> Dict[str, bool]:
        """Start test servers on all ports"""
        results = {}

        for name, port in self.ports.items():
            server = SimpleTestServer(port)
            if server.start():
                self.servers[name] = server
                results[name] = True
                print(f"✅ Test server started on port {port}")
            else:
                results[name] = False
                print(f"❌ Could not start server on port {port} (already in use?)")

        # Give servers time to start
        time.sleep(1)
        return results

    def stop_test_servers(self):
        """Stop all test servers"""
        for server in self.servers.values():
            server.stop()
        self.servers.clear()

    def run_validation(self):
        """Run the complete validation process"""
        print("\n" + "="*60)
        print("🚀 SUBNET 27 PORT VALIDATOR")
        print("="*60)

        # Step 1: Get public IP
        print("\n📡 Getting public IP...")
        self.public_ip = self.get_public_ip()

        if not self.public_ip:
            print("❌ Cannot determine public IP. Are you connected to internet?")
            return False

        print(f"✅ Public IP: {self.public_ip}")

        # Step 2: Check if behind NAT
        print("\n🔍 Checking network type...")
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()

            if local_ip.startswith(('192.168.', '10.', '172.')):
                print(f"⚠️  You're behind NAT (Local IP: {local_ip})")
                print("   You'll need to configure port forwarding on your router!")
            else:
                print(f"✅ Direct connection detected (IP: {local_ip})")
        except Exception:
            print("⚠️  Could not determine network type")

        # Step 3: Check firewall
        print("\n🔒 Checking firewall...")
        fw_status = self.check_firewall()

        if fw_status.get('active'):
            print(f"⚠️  Firewall is active ({fw_status['firewall']})")
            missing_rules = []
            for name, port in self.ports.items():
                if port not in fw_status.get('rules', []):
                    missing_rules.append(f"sudo ufw allow {port}/tcp")

            if missing_rules:
                print("   Missing rules. Run these commands:")
                for cmd in missing_rules:
                    print(f"   $ {cmd}")
        else:
            print("✅ Firewall is not blocking")

        # Step 4: Start test servers
        print("\n🚀 Starting test servers...")
        server_status = self.start_test_servers()

        if not any(server_status.values()):
            print("❌ Could not start any test servers. Ports might be in use.")
            return False

        # Step 5: Test external access
        print(f"\n🌐 Testing external access from {self.public_ip}...")
        print("   (This checks if ports are accessible from the internet)")

        all_good = True
        results = {}

        for name, port in self.ports.items():
            if not server_status.get(name):
                print(f"\n   ❌ Port {port}: Skipped (server not running)")
                results[name] = False
                all_good = False
                continue

            print(f"\n   Testing port {port}...")
            accessible, message = self.check_port_external(port)
            results[name] = accessible

            if accessible:
                print(f"   ✅ Port {port}: ACCESSIBLE from internet")
            else:
                print(f"   ❌ Port {port}: NOT ACCESSIBLE - {message}")
                all_good = False

        # Step 6: Stop servers
        self.stop_test_servers()

        # Final summary
        print("\n" + "="*60)
        print("📊 VALIDATION SUMMARY")
        print("="*60)

        for name, port in self.ports.items():
            status = "✅ PASS" if results.get(name) else "❌ FAIL"
            print(f"  {name}: {status}")

        print("="*60)

        if all_good:
            print("\n🎉 SUCCESS! All ports are accessible from the internet.")
            print("   Your miner is ready for Subnet 27!")
        else:
            print("\n❌ VALIDATION FAILED")
            print("\n📝 Troubleshooting steps:")
            print("   1. If on cloud hosting:")
            print("      - Check your provider's security groups/firewall rules")
            external_ports_str = ', '.join(str(p) for p in self.external_ports)
            print(f"      - Ensure inbound rules allow TCP on ports {self.ssh_port}, {self.axon_port}, {external_ports_str}")
            print("   2. If on home network:")
            print("      - Configure port forwarding on your router")
            print("      - Forward external ports to your machine's local IP")
            print("   3. Check local firewall:")
            print(f"      - Run: sudo ufw allow {self.ssh_port}/tcp")
            print(f"      - Run: sudo ufw allow {self.axon_port}/tcp")
            for port in self.external_ports:
                print(f"      - Run: sudo ufw allow {port}/tcp")

        print("\n" + "="*60)
        return all_good


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Port validator for Subnet 27',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 validate_miner_ports.py                               # Use default ports
  python3 validate_miner_ports.py --ssh-port 2222               # Custom SSH port
  python3 validate_miner_ports.py --ssh-port 4444 --axon-port 8091 --external-ports 27015,27016,27017,27018
        """
    )

    parser.add_argument('--ssh-port', type=int, default=4444,
                       help='SSH port (default: 4444)')
    parser.add_argument('--axon-port', type=int, default=8091,
                       help='Axon port (default: 8091)')
    parser.add_argument('--external-ports', type=str, default='27015,27016,27017,27018',
                       help='Comma-separated list of external ports (default: 27015,27016,27017,27018)')

    args = parser.parse_args()

    # Parse external ports from comma-separated string to list of integers
    external_ports = [int(p.strip()) for p in args.external_ports.split(',')]

    print("\n🔧 Subnet 27 Port Validator")
    print("   This tool checks if your ports are accessible from the internet")
    print(f"   Testing ports: SSH={args.ssh_port}, Axon={args.axon_port}, External={','.join(str(p) for p in external_ports)}")
    print("   Press Ctrl+C to cancel at any time\n")

    validator = PortValidator(
        ssh_port=args.ssh_port,
        axon_port=args.axon_port,
        external_ports=external_ports
    )

    try:
        success = validator.run_validation()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n🛑 Cancelled by user")
        validator.stop_test_servers()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        validator.stop_test_servers()
        sys.exit(1)


if __name__ == "__main__":
    main()
