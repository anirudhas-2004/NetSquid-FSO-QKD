# Filename: key_distribution.py
# Created: 13-10-2025
# Description: Sets up the quantum key distribution network and processes

import warnings
import os
import numpy as np
import pandas as pd
import pydynaa
import netsquid as ns
import netsquid.qubits.ketstates as ks
import json
from tqdm import tqdm
from netsquid.components.models.delaymodels import FibreDelayModel
from channel_errors import FreeSpaceLossModel, BackgroundNoiseModel
from netsquid.components import ClassicalChannel
from netsquid.components.qchannel import QuantumChannel
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.nodes import Network, Node
from netsquid.nodes.connections import DirectConnection
from netsquid.protocols import NodeProtocol, Signals
from netsquid.qubits import operators as ops
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits.qubitapi import create_qubits
from netsquid.qubits.state_sampler import StateSampler
from netsquid.util.datacollector import DataCollector
from netsquid.util.simtools import sim_time
from quick import elliptic_beam_model
warnings.filterwarnings("ignore", category=DeprecationWarning)

OUTPUT_DIR = "qkd_raw_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# std dev = FWHM / 2.355
def get_total_jitter(jitter_components):
    component_sum = 0
    for component in jitter_components:
        component_sum += component**2
    total_jitter_sigma_ns = np.sqrt(component_sum) / 2.355
    return total_jitter_sigma_ns


def generate_triggers(num_events, avg_rate_hz):
    """
    Poisson process i.e. source emission times are exponentially distributed.
    """
    avg_interval_ns = 1e9 / avg_rate_hz  # Hz to ns
    intervals_ns = np.random.exponential(avg_interval_ns, num_events)
    intervals_ns = np.maximum(intervals_ns, 10.0)
    trigger_times = np.cumsum(intervals_ns)
    print(f"Average source emission interval: {np.mean(intervals_ns)/1e3:.1f} μs")
    return trigger_times


def create_qkd_network(source_fidelity_sq,node_distance_a,node_distance_b,
                       loss_array, error_prob):
    """
    Creates QKD network consisting of photon source, nodes, quantum channels and classical channel.
    """
    network = Network("QKD_Network")
    node_a, node_b, source_node = network.add_nodes(["node_A", "node_B", "source_node"])
    state_sampler = StateSampler(
        [ks.b00, ks.s00], probabilities=[source_fidelity_sq, 1 - source_fidelity_sq]
        )
    source = QSource("source", state_sampler=state_sampler, num_ports=2, status=SourceStatus.EXTERNAL)
    
    source_node.add_subcomponent(source)

    node_a.add_subcomponent(QuantumProcessor(
        "alice", num_positions=2, fallback_to_nonphysical=True,
        ))
    node_a.qmemory.ports["qin0"].notify_all_input = True
    
    node_b.add_subcomponent(QuantumProcessor(
        "bob", num_positions=2, fallback_to_nonphysical=True,
        ))
    node_b.qmemory.ports["qin0"].notify_all_input = True

    # classical channel
    total_dist = node_distance_a + node_distance_b
    classical_connection = DirectConnection(
        "CChannelConn_AB",
        ClassicalChannel("CChannel_A->B", length=total_dist,
                         models={"delay_model": FibreDelayModel(c=200e3)}),
        ClassicalChannel("CChannel_B->A", length=total_dist,
                         models={"delay_model": FibreDelayModel(c=200e3)}))
    network.add_connection(node_a, node_b, connection=classical_connection)

    # free space quantum channel
    quantum_channel_sa = QuantumChannel(
        "QChannel_S->A", length=node_distance_a,
        models={"quantum_loss_model": FreeSpaceLossModel(trans_samples = loss_array),
                "quantum_noise_model": BackgroundNoiseModel(error_prob = error_prob), 
                "delay_model": FibreDelayModel(c=299792)})
    port_name_sa, port_name_a = network.add_connection(
        source_node, node_a, channel_to=quantum_channel_sa, label="quantum")
    
    quantum_channel_sb = QuantumChannel(
        "QChannel_S->B", length=node_distance_b,
        models={"quantum_loss_model": FreeSpaceLossModel(trans_samples = loss_array),
                "quantum_noise_model": BackgroundNoiseModel(error_prob = error_prob), 
                "delay_model": FibreDelayModel(c=299792)})
    port_name_sb, port_name_b = network.add_connection(
        source_node, node_b, channel_to=quantum_channel_sb, label="quantum")

    # port setup
    source_node.subcomponents["source"].ports["qout0"].forward_output(
        source_node.ports[port_name_sa])
    source_node.subcomponents["source"].ports["qout1"].forward_output(
        source_node.ports[port_name_sb])
    node_a.ports[port_name_a].forward_input(
        node_a.qmemory.ports["qin0"])
    node_b.ports[port_name_b].forward_input(
        node_b.qmemory.ports["qin0"])
    
    print(f"QKD network created")
    return network


class QKDNodeProtocol(NodeProtocol):
    """
    This protocol runs on nodes Alice and Bob to record detection events with time.
    """
    def __init__(self, node, name, jitter, stop_time=1e9):
        super().__init__(node=node, name=name)
        self.raw_key_count = 0
        self.stop_time = stop_time
        self.stop_ev = pydynaa.EventType("STOP", "Stop QKD")
        self.node._schedule_after(self.stop_time, self.stop_ev)
        self.jitter = jitter

    def run(self):
        click = self.await_port_input(self.node.qmemory.ports["qin0"])
        stop = pydynaa.EventExpression(self, self.stop_ev)
        while True:
            expr = yield click | stop
        
            if expr.second_term.value:
                print(f"{self.name}: {self.raw_key_count} detections. Force stopping.")
                super().stop()

            if expr.first_term.value:
                basis = np.random.randint(0, 2)
                if basis == 0:
                    result = self.node.qmemory.measure(positions=[0], observable=ops.Z, discard=True)[0][0]
                else:
                    result = self.node.qmemory.measure(positions=[0], observable=ops.X, discard=True)[0][0]        
                time = sim_time() + np.random.normal(0, self.jitter)
                data = {
                    "basis": basis,
                    "result": result,
                    "time": time
                    }
                self.raw_key_count += 1
                self.send_signal(Signals.SUCCESS, data)
            

def setup_experiment(num_pairs, avg_pair_rate_hz, 
                     source_fidelity_sq,node_distance_a,node_distance_b,
                     loss_array, error_prob, total_jitter_sigma_ns):
    """
    Setup experiment by instantiating the network, protocols and creating data collectors.
    """
    network = create_qkd_network(source_fidelity_sq,node_distance_a,node_distance_b,
                                 loss_array, error_prob)
    alice = network.get_node("node_A")
    bob = network.get_node("node_B")
    source_node = network.get_node("source_node")
    
    # protocol expected max time 
    gen_time = num_pairs * (1/ avg_pair_rate_hz) * 1e9
    max_time = gen_time + 1e6  # 1 ms extra buffer
    alice_protocol = QKDNodeProtocol(alice, "Alice", total_jitter_sigma_ns, max_time)
    bob_protocol = QKDNodeProtocol(bob, "Bob", total_jitter_sigma_ns, max_time)

    # data collection
    def record_run(evexpr):
        protocol = evexpr.triggered_events[-1].source
        data = protocol.get_signal_result(Signals.SUCCESS)
        return data
    alice_collector = DataCollector(record_run, include_time_stamp=False, include_entity_name=False)
    bob_collector = DataCollector(record_run, include_time_stamp=False, include_entity_name=False)
    alice_collector.collect_on(pydynaa.EventExpression(
        source=alice_protocol, event_type=Signals.SUCCESS.value))
    bob_collector.collect_on(pydynaa.EventExpression(
        source=bob_protocol, event_type=Signals.SUCCESS.value))
    
    return alice_protocol, bob_protocol, source_node, alice_collector, bob_collector

def run_experiment(config):
    """
    Run QKD experiment for a given number of source emissions.
    """
    ns.sim_reset()
    
    # ADD THIS: Get logger
    from qkd_logger import get_logger
    logger = get_logger()

    num_pairs = config['num_pairs']
    avg_pair_rate_hz = config['avg_pair_rate_hz']
    source_fidelity_sq = config['source_fidelity_sq']
    node_distance_a = config['node_distance_a']
    node_distance_b = config['node_distance_b']
    loss_array = config['loss_array']
    error_prob = config['error_prob']
    jitter_components = config['jitter_components']
    total_jitter_sigma_ns = get_total_jitter(jitter_components)
    
    # ADD THIS: Get progress callback
    progress_callback = config.get('progress_callback', None)

    # ADD LOGGING (keep the print too)
    print(f"{ns.sim_time()}: Starting key distribution")
    logger.log_event(
        source='system',
        sim_time=ns.sim_time(),
        event_type='start',
        data={
            'num_pairs': num_pairs,
            'avg_pair_rate_hz': avg_pair_rate_hz,
            'node_distance_a': node_distance_a,
            'node_distance_b': node_distance_b
        },
        message="Starting key distribution"
    )
    
    alice_protocol, bob_protocol, source_node, alice_collector, bob_collector = setup_experiment(
        num_pairs, avg_pair_rate_hz,
        source_fidelity_sq,node_distance_a,node_distance_b,
        loss_array, error_prob, total_jitter_sigma_ns
    )
    alice, bob = alice_protocol.node, bob_protocol.node
    alice_protocol.start()
    bob_protocol.start()

    trigger_times = generate_triggers(num_pairs, avg_pair_rate_hz)
    source = source_node.subcomponents["source"]
    successful_triggers = 0
    two_perc = 2*int(num_pairs/100)
    pbar = tqdm(
        total=50,
        desc="source emissions",
        bar_format='{desc}: {percentage:3.0f}%|{bar:10}| [{elapsed}]',
        ascii='-#',
        leave=True)
    for _, trigger_time in enumerate(trigger_times):
        current_time = sim_time()
        if trigger_time > current_time:
            ns.sim_run(trigger_time)
        source.trigger()
        successful_triggers += 1
        
        # ADD THESE LINES: Update GUI progress bar
        if progress_callback and successful_triggers % two_perc == 0:
            progress_callback(successful_triggers, num_pairs)
            
        if successful_triggers % two_perc == 0:
            pbar.update(1)
    pbar.close()       
    ns.sim_run()
    
    df_alice = alice_collector.dataframe
    df_bob = bob_collector.dataframe
    
    # ADD LOGGING (keep the prints too)
    print(f"{ns.sim_time()}: Finished key distribution")
    print(f"Emitted pairs: {successful_triggers}/{num_pairs}")
    print(f"Alice detections: {len(df_alice)}")
    print(f"Bob detections: {len(df_bob)}")
    logger.log_event(
        source='system',
        sim_time=ns.sim_time(),
        event_type='completion',
        data={
            'emitted_pairs': successful_triggers,
            'requested_pairs': num_pairs,
            'alice_detections': len(df_alice),
            'bob_detections': len(df_bob)
        },
        message=f"Finished key distribution: {successful_triggers}/{num_pairs} pairs, Alice: {len(df_alice)}, Bob: {len(df_bob)} detections"
    )
    print("=="*30)
    
    # sort once
    df_alice = df_alice.sort_values('time').reset_index(drop=True)
    df_bob = df_bob.sort_values('time').reset_index(drop=True)
    alice_file = os.path.join(OUTPUT_DIR, "alice_raw_timetags.csv")
    bob_file = os.path.join(OUTPUT_DIR, "bob_raw_timetags.csv")
    df_alice.to_csv(alice_file, index=False)
    df_bob.to_csv(bob_file, index=False)
    
    # expected delays
    c_fiber = 299792  # km/s free space
    expected_delay_a = node_distance_a / c_fiber * 1e9  # ns
    expected_delay_b = node_distance_b / c_fiber * 1e9  # ns
    expected_diff = expected_delay_b - expected_delay_a
    
    metadata = {
        'alice_distance_km': node_distance_a,
        'bob_distance_km': node_distance_b,
        'generated_pairs': successful_triggers,
        'alice_detections': len(df_alice),
        'bob_detections': len(df_bob),
        'expected_delay_diff_ns': expected_diff,
        'total_jitter_fwhm_ns': total_jitter_sigma_ns * 2.355
    }
    
    metadata_file = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    return alice_collector, bob_collector, alice, bob

if __name__ == "__main__":
    alice_collector, bob_collector, alice, bob = run_experiment()
    if alice_collector is not None and bob_collector is not None:
        print(f"Experiment completed successfully.")
    else:
        print("Experiment failed.")
