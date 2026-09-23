"""Validated transcription preferences shared by the UI and processor."""
import logging

logger = logging.getLogger(__name__)


def supported_compute_types(device="cpu"):
    import ctranslate2
    return sorted(set(ctranslate2.get_supported_compute_types(device, 0)) - {"auto", "default"})


def cpu_compute_type(runtime):
    requested = runtime.get("whisper_compute_type") or "float32"
    supported = supported_compute_types()
    if requested in supported:
        return requested
    logger.warning("Unsupported saved CPU precision %s; using float32", requested)
    if "float32" not in supported:
        raise RuntimeError("This CPU backend does not support the required float32 fallback")
    return "float32"


def provenance_matches(source, runtime):
    """Legacy transcripts are CPU/float32; explicit reuse is handled by the caller."""
    device = runtime.get("whisper_device", "cpu")
    precision = (runtime.get("whisper_cuda_compute_type", "float16") if device == "cuda"
                 else cpu_compute_type(runtime))
    return (source.get("device", "cpu"), source.get("compute_type", "float32")) == (device, precision)
