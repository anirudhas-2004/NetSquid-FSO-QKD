# Filename: channel_errors.py
# Created: 15-10-2025
# Description: NetSquid adapter for custom loss and noise model

import numpy as np
import netsquid as ns
from netsquid.components.models.qerrormodels import QuantumErrorModel
from netsquid.util import simtools

class FreeSpaceLossModel(QuantumErrorModel):
    """
    Empirical loss model that samples transmittance from a provided array of measured T samples.
    For each photon, it draws a new T and applies loss p_loss = 1 - T.
    """
    def __init__(self, trans_samples, rng=None):
        super().__init__()
        self.trans_samples = np.asarray(trans_samples, dtype=float)
        if self.trans_samples.ndim != 1:
            raise ValueError("trans_samples must be a 1D array")
        if np.any((self.trans_samples < 0) | (self.trans_samples > 1)):
            raise ValueError("trans_samples must be in [0,1]")
        self.rng = rng if rng is not None else simtools.get_random_state()

    def error_operation(self, qubits, delta_time=0, **kwargs):
        """Apply loss by sampling a fresh transmittance for each qubit."""
        for idx, qubit in enumerate(qubits):
            if qubit is None:
                continue
            T = self.rng.choice(self.trans_samples)
            p_loss = 1.0 - T
            self.lose_qubit(qubits, idx, p_loss, rng=self.rng)


class BackgroundNoiseModel(QuantumErrorModel):
    """
    Noise model that depolarizes a qubit with given probability using stochastic quantum operations.
    """
    def __init__(self, error_prob=0.5):
        super().__init__()
        self.error_prob = error_prob
    
    def error_operation(self, qubits, delta_time=0, **kwargs):
        for qubit in qubits:
            if qubit is not None:
                ns.qubits.apply_pauli_noise(qubit, p_weights=(1 - self.error_prob, 
                                                              self.error_prob/3, 
                                                              self.error_prob/3, 
                                                              self.error_prob/3))
