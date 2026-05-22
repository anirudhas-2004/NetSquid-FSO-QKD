# Filename: key_sifting.py
# Created: 16-10-2025
# Description: Contains code to perform coincidence detection, basis matching and QBER testing

import warnings
import os
import numpy as np
import pandas as pd
import netsquid as ns
import json
from netsquid.components import ClassicalChannel
from netsquid.nodes import Network, Node
from netsquid.nodes.connections import Connection, DirectConnection
from netsquid.protocols import NodeProtocol, Protocol, Signals

from netsquid.util.datacollector import DataCollector
from netsquid.util.simtools import sim_time
from coincidence import build_coincidence_histogram, calculate_fwhm, detect_coincidences, extract_matching_basis_events
from coincidence import calculate_qber_and_test_security, save_keys
from netsquid.components.component import Message, Port
from qkd_logger import get_logger  # ADD THIS IMPORT

warnings.filterwarnings("ignore", category=DeprecationWarning)

QBER_THRESHOLD = 0.11
TEST_FRACTION = 0.2

class CoincidenceDetection(NodeProtocol):
    def __init__(self, node, port, data_collector, first, name=None):
        name = name
        super().__init__(node=node, name=name)
        self.port = port
        self.data_collector = data_collector
        self.local_df = None
        self.remote_df = None
        self.local_time = None
        self.remote_time = None
        self.first = first
        self.logger = get_logger()  # ADD THIS LINE

    def run(self):
        cchannel_ready = self.await_port_input(self.port)
        self.local_df = self.data_collector.dataframe
        self.local_time = self.local_df["time"].values
        self.port.tx_output(Message(self.local_df[["time", "basis"]], header="time_data"))
        while True:
            yield cchannel_ready
            classical_message = self.port.rx_input(header="time_data")
            if classical_message:
                self.remote_df, = classical_message.items
                self.remote_time = self.remote_df["time"].values
            if self.local_time is not None and self.remote_time is not None:
                yield from self._get_window()

    def _get_window(self):
        if self.first:
            a,b = self.local_time,self.remote_time
        else:
            b,a = self.local_time,self.remote_time
        bin_centers, counts = build_coincidence_histogram(a,b)
        fwhm, _, measured_delay = calculate_fwhm(bin_centers, counts)
        coincidence_window = 1.5
        yield from self._detect_coincidence(coincidence_window, measured_delay)
    
    def _detect_coincidence(self, coincidence_window, measured_delay):
        if not self.first:
            measured_delay *= -1
        coincidences_df = detect_coincidences(self.local_df, self.remote_df, measured_delay, coincidence_window)
        
        # REPLACE print statements with logging
        source_name = self.name.split('_')[0]  # 'alice' or 'bob'
        
        self.logger.log_event(
            source=source_name,
            sim_time=ns.sim_time(),
            event_type='coincidence',
            data={
                'total_coincidences': coincidences_df.shape[0],
                'window_ns': coincidence_window,
                'measured_delay_ns': measured_delay
            },
            message=f"Total Coincidences: {coincidences_df.shape[0]}"
        )
        
        matching_events = extract_matching_basis_events(coincidences_df)
        
        self.logger.log_event(
            source=source_name,
            sim_time=ns.sim_time(),
            event_type='basis_matching',
            data={
                'total_coincidences': coincidences_df.shape[0],
                'matching_basis': len(matching_events)
            },
            message=f"Matching basis: {len(matching_events)}"
        )
        
        yield from self._test_security(matching_events)

    def _check_qber(self, mismatches, total, qber_threshold = QBER_THRESHOLD):
        qber = mismatches / total if total > 0 else 0
        is_secure = qber <= qber_threshold
        return qber, is_secure
    
    def _test_security(self, matching_events):
        if self.first:
            test_indices, local_test_events, key_events = calculate_qber_and_test_security(matching_events, 
                                                                                 test_fraction=TEST_FRACTION)
            self.port.tx_output(Message([test_indices, local_test_events], header="qber_test"))
            yield self.await_port_input(self.port)
            classical_message = self.port.rx_input(header="qber_test")
            remote_test_events, = classical_message.items
            mismatches = sum(local_test_events['local_result'] != remote_test_events['local_result'])
            qber, is_secure = self._check_qber(mismatches, len(local_test_events))
            
            # REPLACE print with logging
            self.logger.log_event(
                source='alice',
                sim_time=ns.sim_time(),
                event_type='qber',
                data={
                    'qber': float(qber),
                    'is_secure': bool(is_secure),
                    'mismatches': int(mismatches),
                    'total_tested': len(local_test_events),
                    'threshold': QBER_THRESHOLD
                },
                message=f"QBER: {qber:.4f}, Secure: {is_secure}"
            )
            
            with open('qkd_raw_data/metadata.json') as f:
                data = json.load(f)
                data['qber'] = qber
                data['is_secure'] = is_secure
            with open('qkd_raw_data/metadata.json', 'w') as f:
                json.dump(data, f, indent=4)
            
            my_key = save_keys("alice.txt", key_events)
            
            # ADD key generation logging
            self.logger.log_event(
                source='alice',
                sim_time=ns.sim_time(),
                event_type='key_generation',
                data={'key_length': len(my_key)},
                message=f"Generated key of length {len(my_key)}"
            )
            
            self.send_signal(Signals.SUCCESS, my_key)
            
        else:
            yield self.await_port_input(self.port)
            classical_message = self.port.rx_input(header="qber_test")
            if classical_message:
                    stuff = classical_message.items
                    test_indices, remote_test_events = stuff
            local_test_events = matching_events.loc[test_indices].copy()
            self.port.tx_output(Message(local_test_events, header="qber_test"))
            key_events = matching_events.drop(test_indices).copy()
            mismatches = sum(remote_test_events['local_result'] != local_test_events['local_result'])
            qber, is_secure = self._check_qber(mismatches, len(local_test_events))
            
            # REPLACE print with logging
            self.logger.log_event(
                source='bob',
                sim_time=ns.sim_time(),
                event_type='qber',
                data={
                    'qber': float(qber),
                    'is_secure': bool(is_secure),
                    'mismatches': int(mismatches),
                    'total_tested': len(local_test_events),
                    'threshold': QBER_THRESHOLD
                },
                message=f"QBER: {qber:.4f}, Secure: {is_secure}"
            )
            
            my_key = save_keys("bob.txt", key_events)
            
            # ADD key generation logging
            self.logger.log_event(
                source='bob',
                sim_time=ns.sim_time(),
                event_type='key_generation',
                data={'key_length': len(my_key)},
                message=f"Generated key of length {len(my_key)}"
            )
            
            self.send_signal(Signals.SUCCESS, my_key)
