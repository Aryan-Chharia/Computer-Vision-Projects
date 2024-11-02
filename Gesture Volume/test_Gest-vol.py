import pytest
from unittest.mock import patch, MagicMock
from GesVol import VolumeController, VolumeControlApp  # Adjust this based on your module name

class TestVolumeController:

    @patch('subprocess.run')
    def test_set_volume_success(self, mock_run):
        controller = VolumeController()
        mock_run.return_value = MagicMock()  # Mock the return value of subprocess.run

        # Test valid volume setting
        result = controller.set_volume(50)
        assert result is True
        mock_run.assert_called_once_with(['amixer', 'set', 'Master', '50%'], check=True, capture_output=True, text=True)

    @patch('subprocess.run')
    def test_set_volume_out_of_bounds(self, mock_run):
        controller = VolumeController()

        # Test volume lower bound
        result = controller.set_volume(-10)
        assert result is True  # Should not raise an error; volume should be clamped to 0
        mock_run.assert_called_with(['amixer', 'set', 'Master', '0%'], check=True, capture_output=True, text=True)

        # Test volume upper bound
        result = controller.set_volume(150)
        assert result is True  # Should not raise an error; volume should be clamped to 100
        mock_run.assert_called_with(['amixer', 'set', 'Master', '100%'], check=True, capture_output=True, text=True)

    @patch('subprocess.run')
    def test_get_current_volume_success(self, mock_run):
        controller = VolumeController()
        mock_run.return_value = MagicMock(stdout="Simple mixer control 'Master',0\n  Limits:  0 - 100\n  Playback channels: Front Left - Front Right\n  Front Left: 75\n  Front Right: 75\n")

        volume = controller.get_current_volume()
        assert volume == 75

    @patch('subprocess.run')
    def test_get_current_volume_error(self, mock_run):
        controller = VolumeController()
        mock_run.side_effect = Exception("Command failed")

        volume = controller.get_current_volume()
        assert volume == 50  # Default volume should be returned on error

class TestVolumeControlApp:

    @patch('gesture_volume.VolumeController')
    @patch('gesture_volume.HandDetector')
    def test_start_tracking(self, mock_detector, mock_controller):
        app = VolumeControlApp(MagicMock())
        app.start_tracking()
        assert app.running is True

    @patch('gesture_volume.VolumeController')
    @patch('gesture_volume.HandDetector')
    def test_stop_tracking(self, mock_detector, mock_controller):
        app = VolumeControlApp(MagicMock())
        app.start_tracking()
        app.stop_tracking()
        assert app.running is False

# Run this module using pytest from the command line
if __name__ == '__main__':
    pytest.main()
