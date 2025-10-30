#!/usr/bin/env python3
"""
Health Check Module

This module handles healthcheck independently from POG (Proof of Generation).
It runs after POG has finished to verify miner connectivity.
"""

import paramiko
import time
import bittensor as bt
import requests

def upload_health_check_script(ssh_client: paramiko.SSHClient, health_check_script_path: str) -> bool:
    """
    Uploads the health check script to the miner using SFTP.

    Args:
        ssh_client (paramiko.SSHClient): SSH client connected to the miner
        health_check_script_path (str): Local path of the health check script

    Returns:
        bool: True if uploaded successfully, False otherwise
    """
    try:
        sftp = ssh_client.open_sftp()
        sftp.put(health_check_script_path, "/tmp/health_check_server.py")
        sftp.chmod("/tmp/health_check_server.py", 0o755)
        sftp.close()
        return True
    except Exception as e:
        bt.logging.debug(f"Failed to upload health check script - SFTP error: {e}")
        return False

def start_health_check_server_background(ssh_client: paramiko.SSHClient, port: int = 27015, timeout: int = 60) -> tuple[bool, paramiko.Channel | None]:
    """
    Starts the health check server using Paramiko channels.

    Args:
        ssh_client (paramiko.SSHClient): SSH client connected to the miner
        port (int): Port for the health check server
        timeout (int): Wait time in seconds (default 60 seconds)

    Returns:
        tuple: (bool, channel) - (True if started successfully, channel object)
    """
    channel = None
    try:
        # Use paramiko transport and channel
        transport = ssh_client.get_transport()
        channel = transport.open_session()

        # Execute the health check server command using channel
        command = f"python3 /tmp/health_check_server.py --port {port} --timeout {timeout}"
        bt.logging.trace(f"Executing remote command: {command}")
        channel.exec_command(command)

        # Check if the channel is still active (server is running)
        if not channel.closed:
            bt.logging.trace("Remote server channel is open.")
            return True, channel
        else:
            # If the channel closed immediately, it means the command failed to start the server.
            # Collect *all* output for debugging this immediate failure.
            stdout_output = ""
            stderr_output = ""
            while channel.recv_ready():
                stdout_output += channel.recv(4096).decode('utf-8', errors='ignore')
            while channel.recv_stderr_ready():
                stderr_output += channel.recv_stderr(4096).decode('utf-8', errors='ignore')

            bt.logging.debug(f"Health check server crashed immediately on startup - check server logs below")
            if stdout_output:
                bt.logging.debug(f"Server stdout: {stdout_output.strip()}")
            if stderr_output:
                bt.logging.debug(f"Server stderr: {stderr_output.strip()}")

            if channel:
                channel.close()
            return False, None

    except Exception as e:
        bt.logging.debug(f"Failed to start health check server - transport error: {e}")
        if channel:
            channel.close()
        return False, None

def read_channel_output(channel: paramiko.Channel, hotkey: str = "") -> None:
    """
    Reads and logs *all available* output from the channel's stdout and stderr streams
    without blocking indefinitely. This function is designed to drain the buffers.

    Args:
        channel: Paramiko channel object
        hotkey (str): Hotkey for logging context
    """
    current_stdout = ""
    current_stderr = ""

    try:
        # Read all available stdout
        while channel.recv_ready():
            current_stdout += channel.recv(4096).decode('utf-8', errors='ignore')

        # Read all available stderr
        while channel.recv_stderr_ready():
            current_stderr += channel.recv_stderr(4096).decode('utf-8', errors='ignore')

        # Log any output found in this cycle
        if current_stdout:
            bt.logging.trace(f"{hotkey}: Health check server stdout: {current_stdout.strip()}")
        if current_stderr:
            bt.logging.trace(f"{hotkey}: Health check server stderr: {current_stderr.strip()}")

    except Exception as e:
        bt.logging.trace(f"{hotkey}: Error reading channel output: {e}")


def wait_for_port_ready(ssh_client: paramiko.SSHClient, port: int = 27015, timeout: int = 30, hotkey: str = "") -> bool:
    """
    Waits for a health endpoint to become available using urllib.request.

    This function checks if the health check server is responding by making
    HTTP requests to the root endpoint (/) on the specified port.

    Args:
        ssh_client (paramiko.SSHClient): SSH client connected to the miner
        port (int): Port to check
        timeout (int): Maximum time to wait in seconds
        hotkey (str): Hotkey for logging context

    Returns:
        bool: True if health endpoint becomes available within timeout, False otherwise
    """
    start_time = time.time()
    check_interval = 1

    while time.time() - start_time < timeout:
        try:
            command = f'python3 -c \'import urllib.request; urllib.request.urlopen("http://127.0.0.1:{port}/", timeout=2)\''
            bt.logging.trace(f"{hotkey}: Checking health endpoint on port {port} (path /)")

            stdin, stdout, stderr = ssh_client.exec_command(command)
            exit_status = stdout.channel.recv_exit_status()

            if exit_status == 0:
                bt.logging.debug(f"{hotkey}: Health endpoint on port {port} (path /) is now responding")
                return True

        except Exception as e:
            bt.logging.trace(f"{hotkey}: Error checking health endpoint: {e}")

        time.sleep(check_interval)

    bt.logging.debug(f"{hotkey}: Health check server not responding on port {port} - server may not be running or port may be blocked by firewall")
    return False

def kill_health_check_server(ssh_client: paramiko.SSHClient, port: int = 27015) -> bool:
    """
    Kills the health check server process using PID file.

    Args:
        ssh_client (paramiko.SSHClient): SSH client connected to the miner
        port (int): Port of the health check server

    Returns:
        bool: True if killed successfully, False otherwise
    """
    try:
        pid_file_path = f"/tmp/health_check_server_{port}.pid"

        # Read PID from file
        stdin, stdout, stderr = ssh_client.exec_command(f"cat {pid_file_path} 2>/dev/null || echo ''")
        pid_output = stdout.read().decode('utf-8').strip()

        if not pid_output:
            bt.logging.trace(f"Health check server PID file not found, server may not be running")
            return True

        try:
            pid = int(pid_output)
        except ValueError:
            bt.logging.trace(f"Invalid PID in file: {pid_output}")
            return True

        # Kill process using PID
        stdin, stdout, stderr = ssh_client.exec_command(f"kill {pid} 2>/dev/null || echo 'Process not found'")
        exit_status = stdout.channel.recv_exit_status()

        if exit_status == 0:
            bt.logging.trace(f"Health check server (PID: {pid}) killed successfully")
            return True
        else:
            bt.logging.trace(f"Health check server (PID: {pid}) was not running or already killed")
            return True

    except Exception as e:
        bt.logging.trace(f"Error killing health check server: {e}")
        return False

def wait_for_health_check(host: str, port: int, timeout: int = 30, retry_interval: int = 1) -> bool:
    """
    Waits for the health check server to be available via HTTP.

    Args:
        host (str): Miner host
        port (int): Health check server port
        timeout (int): Maximum wait time in seconds
        retry_interval (int): Interval between retries in seconds

    Returns:
        bool: True if health check is successful, False otherwise
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://{host}:{port}", timeout=2)

            if response.status_code == 200:
                bt.logging.trace(f"HTTP Health check successful on {host}:{port}!")
                return True
            else:
                bt.logging.debug(f"HTTP Health check server returned status code {response.status_code} instead of 200 on {host}:{port}")
                # Continue to next iteration - don't return here

        except requests.exceptions.ConnectionError as e:
            bt.logging.trace(f"HTTP Connection error to {host}:{port}: {e}")
        except requests.exceptions.Timeout as e:
            bt.logging.trace(f"HTTP Timeout error to {host}:{port}: {e}")
        except requests.exceptions.RequestException as e:
            bt.logging.trace(f"HTTP Request error to {host}:{port}: {e}")

        time.sleep(retry_interval)

    bt.logging.debug(f"External health check failed on {host}:{port} - port may be blocked by firewall or miner is misconfigured")
    return False

def perform_health_check(
    axon: bt.AxonInfo,
    miner_info: dict[str, str | int]
) -> bool:
    """
    Performs health check on a miner after POG has finished.

    Args:
        axon: Axon information of the miner
        miner_info: Miner information (host, port, etc.) - always provided by POG

    Returns:
        bool: True if health check is successful, False otherwise
    """
    hotkey = axon.hotkey
    host: str | None = None
    ssh_client: paramiko.SSHClient | None = None
    channel: paramiko.Channel | None = None

    try:
        host = miner_info['host']

        # Connect via SSH
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            bt.logging.trace(f"{hotkey}: Attempting SSH connection to {host}")
            ssh_client.connect(host, port=miner_info.get('port', 22), username=miner_info['username'], password=miner_info['password'], timeout=10)
            bt.logging.trace(f"{hotkey}: SSH connection successful.")
        except Exception as ssh_error:
            bt.logging.debug(f"{hotkey}: SSH connection failed - miner may be offline or credentials incorrect: {ssh_error}")
            return False

        health_check_script_path = "neurons/Validator/health_check_server.py"

        if not upload_health_check_script(ssh_client, health_check_script_path):
            bt.logging.debug(f"{hotkey}: Failed to upload health check script - miner may have insufficient disk space or permissions")
            return False

        # Get external ports to validate
        external_user_ports = miner_info.get('external_user_ports', {})

        # Start health check servers on all internal ports
        channels = []
        internal_ports = list(external_user_ports.keys())

        for internal_port in internal_ports:
            bt.logging.trace(f"{hotkey}: Starting health check server in background on port {internal_port}.")
            server_started, channel = start_health_check_server_background(ssh_client, int(internal_port), timeout=60)

            if not server_started or channel is None:
                bt.logging.debug(f"{hotkey}: Failed to start health check server on port {internal_port} - miner may have insufficient resources or Python not available")
                # Close any previously opened channels
                for ch in channels:
                    if ch and not ch.closed:
                        ch.close()
                return False

            channels.append(channel)

        server_ready_timeout = 15

        # Wait for all servers to be ready
        for internal_port in internal_ports:
            bt.logging.debug(f"{hotkey}: Attempting to confirm health check server's internal readiness via port check on {internal_port}.")
            if not wait_for_port_ready(ssh_client, int(internal_port), server_ready_timeout, hotkey):
                bt.logging.debug(f"{hotkey}: Health check server on port {internal_port} failed to start properly - server may have crashed or port is blocked")
                # Close all channels
                for ch in channels:
                    if ch and not ch.closed:
                        ch.close()
                return False

        bt.logging.debug(f"{hotkey}: All health check servers confirmed internally ready via port check.")

        health_check_timeout = 15
        health_check_retry_interval = 1

        # Validate all external ports
        all_ports_valid = True
        for internal_port, external_port in external_user_ports.items():
            bt.logging.trace(f"{hotkey}: Performing external HTTP health check on {host}:{external_port} (internal: {internal_port}).")

            health_check_success = wait_for_health_check(
                host,
                external_port,
                timeout=health_check_timeout,
                retry_interval=health_check_retry_interval
            )

            if not health_check_success:
                bt.logging.debug(f"{hotkey}: External health check failed for port {external_port} (internal: {internal_port}) - port may be blocked by firewall or miner is misconfigured")
                all_ports_valid = False
                break
            else:
                bt.logging.trace(f"{hotkey}: Port {external_port} (internal: {internal_port}) validation successful.")

        # Read output from all channels
        for channel in channels:
            if channel and not channel.closed:
                bt.logging.trace(f"{hotkey}: Reading server output from channel.")
                read_channel_output(channel, hotkey)

        if not all_ports_valid:
            # Close all channels before returning
            for channel in channels:
                if channel and not channel.closed:
                    channel.close()
            return False

        # Kill all health check servers
        bt.logging.trace(f"{hotkey}: Health check successful. Attempting to kill all health check servers.")
        for internal_port in internal_ports:
            if kill_health_check_server(ssh_client, int(internal_port)):
                bt.logging.trace(f"{hotkey}: Health check server on port {internal_port} successfully terminated.")
            else:
                bt.logging.debug(f"{hotkey}: Failed to explicitly kill health check server on port {internal_port}, but check was successful. It might be self-terminating.")

        # Read final output and close all channels
        for channel in channels:
            if channel and not channel.closed:
                bt.logging.trace(f"{hotkey}: Reading final server output after termination.")
                read_channel_output(channel, hotkey)
                channel.close()

        return True

    except Exception as e:
        bt.logging.debug(f"{hotkey}: Unexpected error during health check - miner may be in an unstable state: {e}")
        return False

    finally:
        if ssh_client is not None:
            try:
                if channel and not channel.closed:
                    bt.logging.trace(f"{hotkey}: Closing Paramiko channel.")
                    channel.close()
                ssh_client.close()
                bt.logging.trace(f"{hotkey}: SSH connection closed.")
            except Exception as e:
                bt.logging.trace(f"{hotkey}: Error closing SSH connection or channel: {e}")
