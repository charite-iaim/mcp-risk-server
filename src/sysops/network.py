# src/sysops/network.py

import logging
import os

logger = logging.getLogger(__name__)


def set_proxy(cfg):
    """
    Set the HTTP and HTTPS proxy environment variables based on the configuration.

    Args:
        cfg (dict): Configuration dictionary containing network settings.
    """
    if "network" in cfg:
        original_proxies = {}
        original_proxies["HTTP_PROXY"] = os.environ.get("HTTP_PROXY", None)
        original_proxies["HTTPS_PROXY"] = os.environ.get("HTTPS_PROXY", None)

        # Ensure network settings are present
        http_proxy = cfg.get("network", {"http_proxy": None}).get("http_proxy", None)
        https_proxy = cfg.get("network", {"https_proxy": None}).get("https_proxy", None)
        if http_proxy and https_proxy:
            os.environ["HTTP_PROXY"] = http_proxy
            os.environ["HTTPS_PROXY"] = https_proxy
            logger.info(
                f"Network HTTP(S) proxies set to {http_proxy} and {https_proxy}."
            )

        elif http_proxy:
            os.environ["HTTP_PROXY"] = http_proxy
            os.environ["HTTPS_PROXY"] = http_proxy
            logger.info(f"Network proxy set to {http_proxy} for both HTTP and HTTPS.")

        elif https_proxy:
            os.environ["HTTPS_PROXY"] = https_proxy
            os.environ["HTTP_PROXY"] = https_proxy
            logger.info(f"Network proxy set to {https_proxy} for both HTTP and HTTPS.")
        return original_proxies

    logger.info("No network proxy configured.")
    return None


def clear_proxy(proxy_settings=None):
    """
    Restore original HTTP and HTTPS proxy settings.
    """
    if proxy_settings:
        os.environ["HTTP_PROXY"] = proxy_settings["HTTP_PROXY"]
        os.environ["HTTPS_PROXY"] = proxy_settings["HTTPS_PROXY"]

    logger.info("Network proxies reset.")


def collect_api_keys(cfg):
    """
    Collect API keys from the configuration or environment variables.
    Keys defined in the configuration will override those in the environment.

    Args:
        cfg (dict): Configuration dictionary containing API keys.

    Returns:
        dict: A dictionary of API keys.
    """
    api_keys = {
        "api_key": cfg.get("api_key", None),
        "org_key": cfg.get("org_key", None),
        "project_id": cfg.get("project_id", None),
    }
    for key in ["API_KEY", "ORG_KEY", "PROJECT_ID"]:
        if key in os.environ:
            api_keys[key.lower()] = os.environ[key]
    # fetch Perplexity API key if not set yet
    if cfg["provider"] == "perplexity":
        if not api_keys["api_key"] and "PERPLEXITY_API_KEY" in os.environ:
            api_keys["api_key"] = os.environ["PERPLEXITY_API_KEY"]
    return {k: v for k, v in api_keys.items() if v is not None}

