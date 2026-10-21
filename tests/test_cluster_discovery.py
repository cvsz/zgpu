import json
from unittest.mock import MagicMock, patch

from cluster.discovery.scanner import (
    DiscoveredNode,
    NetworkComputeScanner,
)
from cluster.network_pool import NetworkCPUPool


def test_discovered_node_properties():
    node = DiscoveredNode(
        ip="192.168.1.100",
        port=9200,
        node_type="network_cpu",
        status="ready",
        latency_ms=1.5,
        capabilities={"cores": 8},
    )
    assert node.url == "http://192.168.1.100:9200"


def test_scanner_probe_node_success():
    scanner = NetworkComputeScanner()
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"ok": True, "node_type": "network_cpu"}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        node = scanner.probe_node("127.0.0.1", 9200)
        assert node is not None
        assert node.ip == "127.0.0.1"
        assert node.port == 9200
        assert node.node_type == "network_cpu"
        assert node.status == "ready"


def test_scanner_subnet_scan():
    scanner = NetworkComputeScanner(probe_ports=[9200])

    mock_node = DiscoveredNode(
        ip="127.0.0.1",
        port=9200,
        node_type="network_cpu",
        status="ready",
        latency_ms=2.0,
        capabilities={},
    )

    with patch.object(scanner, "probe_node", side_effect=lambda ip, port: mock_node if ip == "127.0.0.1" else None):
        results = scanner.scan_subnet(base_ip_prefix="127.0.0", start_host=1, end_host=2)
        assert len(results) == 1
        assert results[0].ip == "127.0.0.1"


def test_network_pool_auto_discovery():
    pool = NetworkCPUPool()
    mock_nodes = [
        DiscoveredNode("192.168.1.10", 9200, "network_cpu", "ready", 1.2, {}),
        DiscoveredNode("192.168.1.20", 8880, "zgpu", "ready", 0.9, {}),
    ]

    with patch("cluster.discovery.scanner.NetworkComputeScanner.scan_subnet", return_value=mock_nodes):
        count = pool.auto_discover_and_powerup(subnet_prefix="192.168.1", start_host=10, end_host=20)
        assert count == 2
        assert len(pool.nodes) == 2
        assert "network_cpu_192.168.1.10_9200" in pool.nodes
        assert "zgpu_192.168.1.20_8880" in pool.nodes
