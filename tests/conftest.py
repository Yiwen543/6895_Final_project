import sys
from unittest.mock import MagicMock

lgpio_mock = MagicMock()
lgpio_mock.gpiochip_open.return_value = 42
sys.modules['lgpio'] = lgpio_mock
