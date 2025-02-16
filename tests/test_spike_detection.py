import pytest
from grnsuite.spike_detection import detect_spikes

def test_spike_detection():
    test_signal = [0, 1, 5, 1, 0]  # Example signal
    spikes = detect_spikes(test_signal, threshold=3)
    assert len(spikes) == 1  # Expect one spike
