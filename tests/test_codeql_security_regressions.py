"""Security regression tests for assembler parsing and beacon binding."""

from unittest.mock import patch

from cluster.discovery.scanner import NetworkComputeScanner
from isa.assembler import Assembler
from isa.instructions import Instruction, Opcode


def test_assembler_strips_inline_comment_without_regex():
    words = Assembler.assemble("GET.TID R1 # comment\nHALT # trailing")
    assert len(words) == 2
    assert Instruction.decode(words[0]).op == Opcode.GET_TID
    assert Instruction.decode(words[1]).op == Opcode.HALT


def test_assembler_long_untrusted_comment():
    assert len(Assembler.assemble("HALT #" + ("x" * 100000))) == 1


def test_beacon_listener_defaults_to_loopback():
    calls = []

    class FakeSocket:
        def setsockopt(self, *_args):
            pass

        def bind(self, address):
            calls.append(address)

        def settimeout(self, *_args):
            pass

        def recvfrom(self, *_args):
            raise TimeoutError

        def close(self):
            pass

    scanner = NetworkComputeScanner()
    with patch("cluster.discovery.scanner.socket.socket", return_value=FakeSocket()):
        assert scanner.listen_beacons(duration_seconds=0) == []
    assert calls == [("127.0.0.1", 9998)]


def test_beacon_listener_explicit_lan_binding():
    bound = []

    class FakeSocket:
        def setsockopt(self, *_args):
            pass

        def bind(self, address):
            bound.append(address)

        def settimeout(self, *_args):
            pass

        def close(self):
            pass

    with patch("cluster.discovery.scanner.socket.socket", return_value=FakeSocket()):
        NetworkComputeScanner().listen_beacons(duration_seconds=0, bind_host="192.0.2.5")
    assert bound == [("192.0.2.5", 9998)]
