import pytest
from message_queue import *


def test_stub_func_without_input():
    with pytest.raises(TypeError):
        do_something()
