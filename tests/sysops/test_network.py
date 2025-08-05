# mcp/tests/sysops/test_network.py

import os
import pytest
from src.sysops.network import set_proxy

def clear_env(monkeypatch):
    for var in ["HTTP_PROXY", "HTTPS_PROXY"]:
        monkeypatch.delenv(var, raising=False)

def test_set_proxy_both(monkeypatch, caplog):
    clear_env(monkeypatch)
    http_proxy = "http://proxy.foo.com:8000"
    https_proxy = "https://proxy.foo.com:8000"
    cfg = {
        "network": {
            "http_proxy": http_proxy, 
            "https_proxy": https_proxy
        }
    }
    with caplog.at_level("INFO"):
        set_proxy(cfg)
    assert os.environ["HTTP_PROXY"] == http_proxy
    assert os.environ["HTTPS_PROXY"] == https_proxy
    logtext = f"Network HTTP(S) proxies set to {http_proxy} and {https_proxy}."
    assert logtext in caplog.text

def test_set_proxy_http_only(monkeypatch, caplog):
    clear_env(monkeypatch)
    http_proxy = "http://proxy.foo.com:8000"
    cfg = {"network": {"http_proxy": http_proxy}}
    with caplog.at_level("INFO"):
        set_proxy(cfg)
    assert os.environ["HTTP_PROXY"] == http_proxy
    assert os.environ["HTTPS_PROXY"] == http_proxy
    logtext = f"Network proxy set to {http_proxy} for both HTTP and HTTPS."
    assert logtext in caplog.text

def test_set_proxy_https_only(monkeypatch, caplog):
    clear_env(monkeypatch)
    https_proxy = "https://proxy.foo.com:8080"
    cfg = {"network": {"https_proxy": https_proxy}}
    with caplog.at_level("INFO"):
        set_proxy(cfg)
    assert os.environ["HTTP_PROXY"] == https_proxy
    assert os.environ["HTTPS_PROXY"] == https_proxy
    logtext = f"Network proxy set to {https_proxy} for both HTTP and HTTPS."
    assert logtext in caplog.text

def test_set_proxy_none(monkeypatch, caplog):
    clear_env(monkeypatch)
    cfg = {}
    with caplog.at_level("INFO"):
        set_proxy(cfg)
    assert "No network proxy configured" in caplog.text
    assert "HTTP_PROXY" not in os.environ
    assert "HTTPS_PROXY" not in os.environ