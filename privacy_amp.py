# Filename: privacy_amp.py
# Created: 22-10-2025
# Description: NetSquid adapter for randextract toeplitz hashing
# uses: https://randextract.crypto-lab.ch/

import warnings
import json
import numpy as np
import netsquid as ns
from galois import GF2
from randextract.src.randextract.toeplitz_hashing import ToeplitzHashing
from randextract.src.randextract.randomness_extractor import RandomnessExtractor
from netsquid.protocols import NodeProtocol, Signals
from netsquid.components.component import Message

warnings.filterwarnings("ignore", category=DeprecationWarning)

def read_key(file_path):
    """Read a .txt file as a GF2 binary array."""
    with open(file_path, 'r') as f:
        key_str = f.read().strip()
    return GF2([int(bit) for bit in key_str])

def save_key(file_path, key_array):
    """Save GF2 array as a binary string to .txt."""
    key_str = ''.join(str(bit) for bit in key_array)
    with open(file_path, 'w') as f:
        f.write(key_str)

class PrivacyAmplificationServer(NodeProtocol):
    """Alice - generates seed and performs privacy amplification"""
    
    def __init__(self, node, port, alice_key_file="alice_recon.txt", 
                 output_file="alice_amplified.txt", name=None):
        super().__init__(node=node, name=name)
        self.port = port
        self.alice_key_file = alice_key_file
        self.output_file = output_file
        self.extractor = None
        self.seed = None
        
    def run(self):
        # Load reconciliation stats
        with open("reconciliation_stats.json", "r") as f:
            data = json.load(f)
        
        key_size = data["key_size"]
        reply_parity_bits = data["reply_parity_bits"]
        
        # Calculate parameters
        input_length = key_size
        leaked_bits = reply_parity_bits
        relative_source_entropy = (input_length - leaked_bits) / input_length
        error_bound = 1e-6
        
        # Calculate optimal output length
        optimal_output_length = ToeplitzHashing.calculate_length(
            extractor_type="quantum",
            input_length=input_length,
            relative_source_entropy=relative_source_entropy,
            error_bound=error_bound
        )
        print(f"Optimal output length: {optimal_output_length} bits (entropy={relative_source_entropy:.2f}, ε={error_bound})")
        
        # Create extractor
        self.extractor = RandomnessExtractor.create(
            extractor_type="toeplitz",
            input_length=input_length,
            output_length=optimal_output_length
        )
        #print(f"Seed length required: {self.extractor.seed_length} bits")
        
        # Generate random seed
        self.seed = GF2.Random(self.extractor.seed_length)
        
        # Send seed to Bob
        seed_list = self.seed.tolist()
        self.port.tx_output(Message(seed_list, header="seed"))
        #print(f"Alice sent seed ({len(seed_list)} bits) @ {ns.sim_time()}")
        
        # Perform extraction
        alice_key = read_key(self.alice_key_file)
        alice_extracted = self.extractor.extract(alice_key, self.seed)
        save_key(self.output_file, alice_extracted)
        #print(f"Alice's amplified key saved to {self.output_file} ({len(alice_extracted)} bits)")
        
        # Signal completion
        self.send_signal(Signals.SUCCESS, alice_extracted)

class PrivacyAmplificationClient(NodeProtocol):
    """Bob - receives seed and performs privacy amplification"""
    
    def __init__(self, node, port, bob_key_file="bob_recon.txt", 
                 output_file="bob_amplified.txt", name=None):
        super().__init__(node=node, name=name)
        self.port = port
        self.bob_key_file = bob_key_file
        self.output_file = output_file
        self.extractor = None
        self.seed = None
        
    def run(self):
        # Wait for seed from Alice
        cchannel_ready = self.await_port_input(self.port)
        yield cchannel_ready
        
        classical_message = self.port.rx_input(header="seed")
        if classical_message:
            seed_list = classical_message.items
            self.seed = GF2(seed_list)
            #print(f"Bob received seed ({len(seed_list)} bits) @ {ns.sim_time()}")
        
        # Load reconciliation stats (same as Alice)
        with open("reconciliation_stats.json", "r") as f:
            data = json.load(f)
        
        key_size = data["key_size"]
        reply_parity_bits = data["reply_parity_bits"]
        
        # Calculate parameters (same as Alice)
        input_length = key_size
        leaked_bits = reply_parity_bits
        relative_source_entropy = (input_length - leaked_bits) / input_length
        error_bound = 1e-6
        
        # Calculate optimal output length (same as Alice)
        optimal_output_length = ToeplitzHashing.calculate_length(
            extractor_type="quantum",
            input_length=input_length,
            relative_source_entropy=relative_source_entropy,
            error_bound=error_bound
        )
        
        # Create extractor (same as Alice)
        self.extractor = RandomnessExtractor.create(
            extractor_type="toeplitz",
            input_length=input_length,
            output_length=optimal_output_length
        )
        
        # Perform extraction using received seed
        bob_key = read_key(self.bob_key_file)
        bob_extracted = self.extractor.extract(bob_key, self.seed)
        save_key(self.output_file, bob_extracted)
        #print(f"Bob's amplified key saved to {self.output_file} ({len(bob_extracted)} bits)")
        
        # Signal completion
        self.send_signal(Signals.SUCCESS, bob_extracted)
