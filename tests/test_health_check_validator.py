import pytest
from unittest import mock

# Import all health check functions at module level
from neurons.Validator.health_check import (
    upload_health_check_script,
    start_health_check_server_background,
    read_channel_output,
    wait_for_port_ready,
    kill_health_check_server,
    perform_health_check
)

# --- Fixtures for common objects ---

@pytest.fixture
def mock_ssh_client():
    """Returns a mock SSH client."""
    mock_ssh = mock.MagicMock()
    mock_ssh.connect.return_value = None
    return mock_ssh


@pytest.fixture
def mock_channel():
    """Returns a mock channel."""
    mock_channel = mock.MagicMock()
    mock_channel.closed = False
    mock_channel.recv_ready.return_value = False
    mock_channel.recv_stderr_ready.return_value = False
    return mock_channel


@pytest.fixture
def mock_transport():
    """Returns a mock transport."""
    mock_transport = mock.MagicMock()
    return mock_transport


@pytest.fixture
def mock_axon():
    """Returns a mock axon."""
    axon = mock.MagicMock()
    axon.hotkey = "test_hotkey"
    return axon


@pytest.fixture
def miner_info():
    """Returns test miner info."""
    return {
        'host': 'localhost',
        'port': 22,
        'username': 'test',
        'password': 'test',
        'external_user_ports': {27015: 8000}  # internal_port: external_port mapping
    }


@pytest.fixture
def mock_health_check_functions():
    """Mocks all health check functions."""
    mock_upload_script = mock.MagicMock(return_value=True)
    mock_start_server = mock.MagicMock(return_value=(True, mock.MagicMock()))
    mock_wait_port_ready = mock.MagicMock(return_value=True)
    mock_wait_health = mock.MagicMock(return_value=True)
    mock_kill_server = mock.MagicMock(return_value=True)

    patcher1 = mock.patch('neurons.Validator.health_check.upload_health_check_script', mock_upload_script)
    patcher3 = mock.patch('neurons.Validator.health_check.start_health_check_server_background', mock_start_server)
    patcher4 = mock.patch('neurons.Validator.health_check.wait_for_port_ready', mock_wait_port_ready)
    patcher5 = mock.patch('neurons.Validator.health_check.wait_for_health_check', mock_wait_health)
    patcher6 = mock.patch('neurons.Validator.health_check.kill_health_check_server', mock_kill_server)
    patcher7 = mock.patch('time.sleep', return_value=None)

    patcher1.start()
    patcher3.start()
    patcher4.start()
    patcher5.start()
    patcher6.start()
    patcher7.start()

    yield {
        'upload_script': mock_upload_script,
        'start_server': mock_start_server,
        'wait_port_ready': mock_wait_port_ready,
        'wait_health': mock_wait_health,
        'kill_server': mock_kill_server
    }

    patcher7.stop()
    patcher6.stop()
    patcher5.stop()
    patcher4.stop()
    patcher3.stop()
    patcher1.stop()


@pytest.fixture
def mock_paramiko():
    """Mocks paramiko SSH client."""
    mock_ssh = mock.MagicMock()
    mock_ssh.connect.return_value = None

    patcher = mock.patch('paramiko.SSHClient', return_value=mock_ssh)
    patcher.start()

    yield mock_ssh

    patcher.stop()


# ============================================================================
# TESTS FOR VALIDATOR HEALTH CHECK (SSH and HTTP Client Component)
# ============================================================================

class TestValidatorHealthCheck:
    """Tests for the validator health check component (SSH and HTTP client)."""

    def test_upload_health_check_script_success(self, mock_ssh_client):
        """Test successful script upload."""
        mock_sftp = mock.MagicMock()
        mock_ssh_client.open_sftp.return_value = mock_sftp

        result = upload_health_check_script(mock_ssh_client, "test_script.py")

        assert result is True
        mock_sftp.put.assert_called_once_with("test_script.py", "/tmp/health_check_server.py")
        mock_sftp.chmod.assert_called_once_with("/tmp/health_check_server.py", 0o755)
        mock_sftp.close.assert_called_once()

    def test_upload_health_check_script_failure(self, mock_ssh_client):
        """Test script upload failure."""
        mock_ssh_client.open_sftp.side_effect = Exception("SFTP error")

        result = upload_health_check_script(mock_ssh_client, "test_script.py")

        assert result is False

    def test_start_health_check_server_background_success(self, mock_ssh_client, mock_channel, mock_transport):
        """Test successful background server start."""
        mock_transport.open_session.return_value = mock_channel
        mock_ssh_client.get_transport.return_value = mock_transport

        result, channel = start_health_check_server_background(mock_ssh_client, 8080)

        assert result is True
        assert channel == mock_channel
        mock_channel.exec_command.assert_called_once_with("python3 /tmp/health_check_server.py --port 8080 --timeout 60")

    def test_start_health_check_server_background_channel_closed(self, mock_ssh_client, mock_channel, mock_transport):
        """Test server start with closed channel."""
        mock_channel.closed = True
        mock_channel.recv_ready.return_value = False
        mock_channel.recv_stderr_ready.return_value = False
        mock_transport.open_session.return_value = mock_channel
        mock_ssh_client.get_transport.return_value = mock_transport

        result, channel = start_health_check_server_background(mock_ssh_client, 8080)

        assert result is False
        assert channel is None
        mock_channel.close.assert_called_once()

    def test_start_health_check_server_background_channel_closed_with_output(self, mock_ssh_client, mock_channel, mock_transport):
        """Test server start with closed channel that has output."""
        mock_channel.closed = True
        mock_channel.recv_ready.side_effect = [True, False]
        mock_channel.recv_stderr_ready.side_effect = [True, False]
        mock_channel.recv.return_value = b"error output"
        mock_channel.recv_stderr.return_value = b"stderr output"
        mock_transport.open_session.return_value = mock_channel
        mock_ssh_client.get_transport.return_value = mock_transport

        result, channel = start_health_check_server_background(mock_ssh_client, 8080)

        assert result is False
        assert channel is None
        mock_channel.close.assert_called_once()

    def test_start_health_check_server_background_exception(self, mock_ssh_client):
        """Test server start with exception."""
        mock_ssh_client.get_transport.side_effect = Exception("Transport error")

        result, channel = start_health_check_server_background(mock_ssh_client, 8080)

        assert result is False
        assert channel is None

    def test_start_health_check_server_background_exception_in_finally(self, mock_ssh_client):
        """Test start_health_check_server_background with exception in finally block."""
        mock_ssh_client.get_transport.side_effect = Exception("Transport error")

        result, channel = start_health_check_server_background(mock_ssh_client, 8080)

        assert result is False
        assert channel is None

    def test_read_channel_output_success(self, mock_channel):
        """Test successful channel output reading."""
        read_channel_output(mock_channel, "test_hotkey")

        mock_channel.recv_ready.assert_called()
        mock_channel.recv_stderr_ready.assert_called()

    def test_read_channel_output_with_data(self, mock_channel):
        """Test channel output reading with actual data."""
        mock_channel.recv_ready.side_effect = [True, False]
        mock_channel.recv_stderr_ready.side_effect = [True, False]
        mock_channel.recv.return_value = b"test output"
        mock_channel.recv_stderr.return_value = b"error output"

        read_channel_output(mock_channel, "test_hotkey")

        mock_channel.recv.assert_called()
        mock_channel.recv_stderr.assert_called()

    def test_read_channel_output_exception(self, mock_channel):
        """Test channel output reading with exception."""
        mock_channel.recv_ready.side_effect = [True, False]
        mock_channel.recv.side_effect = Exception("Channel error")

        read_channel_output(mock_channel, "test_hotkey")

    def test_kill_health_check_server_success(self, mock_ssh_client):
        """Test successful server kill using PID file."""
        # Mock first call: read PID file
        mock_stdin1 = mock.MagicMock()
        mock_stdout1 = mock.MagicMock()
        mock_stderr1 = mock.MagicMock()
        mock_stdout1.read.return_value = b"12345"
        mock_stdout1.channel.recv_exit_status.return_value = 0

        # Mock second call: kill process
        mock_stdin2 = mock.MagicMock()
        mock_stdout2 = mock.MagicMock()
        mock_stderr2 = mock.MagicMock()
        mock_stdout2.channel.recv_exit_status.return_value = 0

        mock_ssh_client.exec_command.side_effect = [
            (mock_stdin1, mock_stdout1, mock_stderr1),
            (mock_stdin2, mock_stdout2, mock_stderr2)
        ]

        result = kill_health_check_server(mock_ssh_client, 8080)

        assert result is True
        assert mock_ssh_client.exec_command.call_count == 2

    def test_kill_health_check_server_not_running(self, mock_ssh_client):
        """Test server kill when server is not running."""
        # Mock first call: read PID file (empty)
        mock_stdin1 = mock.MagicMock()
        mock_stdout1 = mock.MagicMock()
        mock_stderr1 = mock.MagicMock()
        mock_stdout1.read.return_value = b""
        mock_stdout1.channel.recv_exit_status.return_value = 0

        mock_ssh_client.exec_command.return_value = (mock_stdin1, mock_stdout1, mock_stderr1)

        result = kill_health_check_server(mock_ssh_client, 8080)

        assert result is True
        mock_ssh_client.exec_command.assert_called_once()

    def test_kill_health_check_server_invalid_pid(self, mock_ssh_client):
        """Test server kill with invalid PID in file."""
        # Mock first call: read PID file (invalid)
        mock_stdin1 = mock.MagicMock()
        mock_stdout1 = mock.MagicMock()
        mock_stderr1 = mock.MagicMock()
        mock_stdout1.read.return_value = b"invalid_pid"
        mock_stdout1.channel.recv_exit_status.return_value = 0

        mock_ssh_client.exec_command.return_value = (mock_stdin1, mock_stdout1, mock_stderr1)

        result = kill_health_check_server(mock_ssh_client, 8080)

        assert result is True
        mock_ssh_client.exec_command.assert_called_once()

    def test_kill_health_check_server_exception(self, mock_ssh_client):
        """Test server kill with exception."""
        mock_ssh_client.exec_command.side_effect = Exception("SSH error")

        result = kill_health_check_server(mock_ssh_client, 8080)

        assert result is False

    def test_wait_for_port_ready_success(self, mock_ssh_client):
        """Test successful port readiness check."""
        # Mock successful command execution
        mock_stdin = mock.MagicMock()
        mock_stdout = mock.MagicMock()
        mock_stderr = mock.MagicMock()
        mock_stdout.channel.recv_exit_status.return_value = 0

        mock_ssh_client.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)

        result = wait_for_port_ready(mock_ssh_client, 8080, timeout=1)

        assert result is True
        mock_ssh_client.exec_command.assert_called()

    def test_wait_for_port_ready_timeout(self, mock_ssh_client):
        """Test port readiness check with timeout."""
        # Mock failed command execution
        mock_stdin = mock.MagicMock()
        mock_stdout = mock.MagicMock()
        mock_stderr = mock.MagicMock()
        mock_stdout.channel.recv_exit_status.return_value = 1

        mock_ssh_client.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)

        result = wait_for_port_ready(mock_ssh_client, 8080, timeout=1)

        assert result is False

    def test_wait_for_port_ready_exception(self, mock_ssh_client):
        """Test port readiness check with exception."""
        mock_ssh_client.exec_command.side_effect = Exception("SSH error")

        result = wait_for_port_ready(mock_ssh_client, 8080, timeout=1)

        assert result is False


# ============================================================================
# TESTS FOR HEALTH CHECK INTEGRATION (Complete Flow)
# ============================================================================

class TestHealthCheckIntegration:
    """Integration tests for complete health check flow."""

    def test_perform_health_check_success(self, mock_health_check_functions, mock_paramiko, mock_axon, miner_info):
        """Test successful health check execution."""
        result = perform_health_check(mock_axon, miner_info)

        assert result is True
        mock_health_check_functions['upload_script'].assert_called_once()
        # With multiple ports support, these are called once per port
        num_ports = len(miner_info['external_user_ports'])
        assert mock_health_check_functions['start_server'].call_count == num_ports
        assert mock_health_check_functions['wait_port_ready'].call_count == num_ports
        assert mock_health_check_functions['wait_health'].call_count == num_ports
        assert mock_health_check_functions['kill_server'].call_count == num_ports

    def test_perform_health_check_ssh_failure(self, mock_paramiko, mock_axon, miner_info):
        """Test health check failure when SSH connection fails."""
        mock_paramiko.connect.side_effect = Exception("SSH connection failed")

        result = perform_health_check(mock_axon, miner_info)

        assert result is False

    def test_perform_health_check_server_start_failure(self, mock_paramiko, mock_axon, miner_info):
        """Test health check failure when server fails to start."""
        mock_start_server = mock.MagicMock(return_value=(False, None))
        mock_upload_script = mock.MagicMock(return_value=True)

        with mock.patch('neurons.Validator.health_check.upload_health_check_script', mock_upload_script), \
             mock.patch('neurons.Validator.health_check.start_health_check_server_background', mock_start_server), \
             mock.patch('time.sleep', return_value=None):

            result = perform_health_check(mock_axon, miner_info)

            assert result is False
            mock_upload_script.assert_called_once()
            mock_start_server.assert_called_once()  # Only called once before failing

    def test_perform_health_check_upload_failure(self, mock_paramiko, mock_axon, miner_info):
        """Test health check failure when script upload fails."""
        mock_upload_script = mock.MagicMock(return_value=False)

        with mock.patch('neurons.Validator.health_check.upload_health_check_script', mock_upload_script), \
             mock.patch('time.sleep', return_value=None):

            result = perform_health_check(mock_axon, miner_info)

            assert result is False

    def test_perform_health_check_exception(self, mock_paramiko, mock_axon, miner_info):
        """Test health check with unexpected exception."""
        mock_upload_script = mock.MagicMock(side_effect=Exception("Unexpected error"))

        with mock.patch('neurons.Validator.health_check.upload_health_check_script', mock_upload_script), \
             mock.patch('time.sleep', return_value=None):

            result = perform_health_check(mock_axon, miner_info)

            assert result is False

    def test_perform_health_check_server_not_ready(self, mock_paramiko, mock_axon, miner_info):
        """Test health check when server doesn't signal readiness."""
        mock_upload_script = mock.MagicMock(return_value=True)
        mock_channel = mock.MagicMock()
        mock_channel.closed = False
        mock_channel.recv_ready.return_value = False
        mock_channel.recv_stderr_ready.return_value = False
        mock_start_server = mock.MagicMock(return_value=(True, mock_channel))
        mock_wait_port_ready = mock.MagicMock(return_value=False)

        with mock.patch('neurons.Validator.health_check.upload_health_check_script', mock_upload_script), \
             mock.patch('neurons.Validator.health_check.start_health_check_server_background', mock_start_server), \
             mock.patch('neurons.Validator.health_check.wait_for_port_ready', mock_wait_port_ready), \
             mock.patch('time.sleep', return_value=None):

            result = perform_health_check(mock_axon, miner_info)

            assert result is False
            mock_upload_script.assert_called_once()
            # Should be called once per port
            assert mock_start_server.call_count == len(miner_info['external_user_ports'])
            # Should be called once (fails on first port check)
            mock_wait_port_ready.assert_called_once()

    def test_perform_health_check_http_check_failure(self, mock_paramiko, mock_axon, miner_info):
        """Test health check when HTTP health check fails."""
        mock_upload_script = mock.MagicMock(return_value=True)
        mock_channel = mock.MagicMock()
        mock_channel.closed = False
        mock_channel.recv_ready.return_value = False  # No data to read
        mock_channel.recv_stderr_ready.return_value = False  # No stderr data
        mock_start_server = mock.MagicMock(return_value=(True, mock_channel))
        mock_wait_port_ready = mock.MagicMock(return_value=True)
        mock_wait_health = mock.MagicMock(return_value=False)
        mock_kill_server = mock.MagicMock(return_value=True)

        with mock.patch('neurons.Validator.health_check.upload_health_check_script', mock_upload_script), \
             mock.patch('neurons.Validator.health_check.start_health_check_server_background', mock_start_server), \
             mock.patch('neurons.Validator.health_check.wait_for_port_ready', mock_wait_port_ready), \
             mock.patch('neurons.Validator.health_check.wait_for_health_check', mock_wait_health), \
             mock.patch('neurons.Validator.health_check.kill_health_check_server', mock_kill_server), \
             mock.patch('time.sleep', return_value=None):

            result = perform_health_check(mock_axon, miner_info)

            assert result is False
            mock_upload_script.assert_called_once()
            # All servers should start
            num_ports = len(miner_info['external_user_ports'])
            assert mock_start_server.call_count == num_ports
            assert mock_wait_port_ready.call_count == num_ports
            # HTTP check fails on first port
            mock_wait_health.assert_called_once()

    def test_perform_health_check_kill_server_failure(self, mock_paramiko, mock_axon, miner_info):
        """Test health check when killing server fails but health check succeeds."""
        mock_upload_script = mock.MagicMock(return_value=True)
        mock_start_server = mock.MagicMock(return_value=(True, mock.MagicMock()))
        mock_wait_port_ready = mock.MagicMock(return_value=True)
        mock_wait_health = mock.MagicMock(return_value=True)
        mock_kill_server = mock.MagicMock(return_value=False)

        with mock.patch('neurons.Validator.health_check.upload_health_check_script', mock_upload_script), \
             mock.patch('neurons.Validator.health_check.start_health_check_server_background', mock_start_server), \
             mock.patch('neurons.Validator.health_check.wait_for_port_ready', mock_wait_port_ready), \
             mock.patch('neurons.Validator.health_check.wait_for_health_check', mock_wait_health), \
             mock.patch('neurons.Validator.health_check.kill_health_check_server', mock_kill_server), \
             mock.patch('time.sleep', return_value=None):

            result = perform_health_check(mock_axon, miner_info)

            assert result is True

    def test_perform_health_check_unexpected_exception(self, mock_paramiko, mock_axon, miner_info):
        """Test health check with unexpected exception in main flow."""
        mock_upload_script = mock.MagicMock(side_effect=Exception("Unexpected error in upload"))

        with mock.patch('neurons.Validator.health_check.upload_health_check_script', mock_upload_script), \
             mock.patch('time.sleep', return_value=None):

            result = perform_health_check(mock_axon, miner_info)

            assert result is False

    def test_perform_health_check_channel_output_reading(self, mock_paramiko, mock_axon, miner_info):
        """Test health check with channel output reading after HTTP check."""
        mock_channel = mock.MagicMock()
        mock_channel.closed = False
        mock_channel.recv_ready.return_value = False
        mock_channel.recv_stderr_ready.return_value = False

        mock_upload_script = mock.MagicMock(return_value=True)
        mock_start_server = mock.MagicMock(return_value=(True, mock_channel))
        mock_wait_port_ready = mock.MagicMock(return_value=True)
        mock_wait_health = mock.MagicMock(return_value=True)
        mock_kill_server = mock.MagicMock(return_value=True)
        mock_read_output = mock.MagicMock()

        with mock.patch('neurons.Validator.health_check.upload_health_check_script', mock_upload_script), \
             mock.patch('neurons.Validator.health_check.start_health_check_server_background', mock_start_server), \
             mock.patch('neurons.Validator.health_check.wait_for_port_ready', mock_wait_port_ready), \
             mock.patch('neurons.Validator.health_check.wait_for_health_check', mock_wait_health), \
             mock.patch('neurons.Validator.health_check.kill_health_check_server', mock_kill_server), \
             mock.patch('neurons.Validator.health_check.read_channel_output', mock_read_output), \
             mock.patch('time.sleep', return_value=None):

            result = perform_health_check(mock_axon, miner_info)

            assert result is True
            # read_channel_output is called twice per channel (before and after kill)
            num_ports = len(miner_info['external_user_ports'])
            assert mock_read_output.call_count == num_ports * 2

    def test_perform_health_check_channel_closed_after_http_check(self, mock_paramiko, mock_axon, miner_info):
        """Test health check when channel is closed after HTTP check."""
        mock_channel = mock.MagicMock()
        mock_channel.closed = True

        mock_upload_script = mock.MagicMock(return_value=True)
        mock_start_server = mock.MagicMock(return_value=(True, mock_channel))
        mock_wait_port_ready = mock.MagicMock(return_value=True)
        mock_wait_health = mock.MagicMock(return_value=True)
        mock_kill_server = mock.MagicMock(return_value=True)

        with mock.patch('neurons.Validator.health_check.upload_health_check_script', mock_upload_script), \
             mock.patch('neurons.Validator.health_check.start_health_check_server_background', mock_start_server), \
             mock.patch('neurons.Validator.health_check.wait_for_port_ready', mock_wait_port_ready), \
             mock.patch('neurons.Validator.health_check.wait_for_health_check', mock_wait_health), \
             mock.patch('neurons.Validator.health_check.kill_health_check_server', mock_kill_server), \
             mock.patch('time.sleep', return_value=None):

            result = perform_health_check(mock_axon, miner_info)

            assert result is True

    def test_perform_health_check_channel_output_after_kill(self, mock_paramiko, mock_axon, miner_info):
        """Test health check with channel output reading after server kill."""
        mock_channel = mock.MagicMock()
        mock_channel.closed = False
        mock_channel.recv_ready.return_value = False
        mock_channel.recv_stderr_ready.return_value = False

        mock_upload_script = mock.MagicMock(return_value=True)
        mock_start_server = mock.MagicMock(return_value=(True, mock_channel))
        mock_wait_port_ready = mock.MagicMock(return_value=True)
        mock_wait_health = mock.MagicMock(return_value=True)
        mock_kill_server = mock.MagicMock(return_value=True)
        mock_read_output = mock.MagicMock()

        with mock.patch('neurons.Validator.health_check.upload_health_check_script', mock_upload_script), \
             mock.patch('neurons.Validator.health_check.start_health_check_server_background', mock_start_server), \
             mock.patch('neurons.Validator.health_check.wait_for_port_ready', mock_wait_port_ready), \
             mock.patch('neurons.Validator.health_check.wait_for_health_check', mock_wait_health), \
             mock.patch('neurons.Validator.health_check.kill_health_check_server', mock_kill_server), \
             mock.patch('neurons.Validator.health_check.read_channel_output', mock_read_output), \
             mock.patch('time.sleep', return_value=None):

            result = perform_health_check(mock_axon, miner_info)

            assert result is True
