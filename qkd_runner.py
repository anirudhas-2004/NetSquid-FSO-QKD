# Filename: qkd_runner.py
# Created: 15-10-2025
# Description: Runs the complete simulation and collects all metrics
# This script is used in master_data.ipynb

import json
import numpy as np
import netsquid as ns
from key_distribution import run_experiment
from key_sifting import CoincidenceDetection
from info_reco import CascadeClient, CascadeServer
from privacy_amp import PrivacyAmplificationServer, PrivacyAmplificationClient, read_key
from quick import elliptic_beam_model
from qkd_logger import reset_logger


def calculate_error_prob(condition_type):
    """Calculate error probability for day/night conditions"""
    h = 6.62607015e-34
    c = 299792458
    Hb_night = 1.5e-6  # W m^-2 sr^-1 nm^-1
    Hb_day = 1.5e-3
    Omega = (100e-6)**2  # sr
    Bf = 1.0  # nm
    Delta_t = 1e-9  # s
    a = 0.55
    wavelength_nm = 785
    
    def Nbg(Hb, Omega, Bf, Delta_t, a, wavelength_nm):
        A_rx = np.pi*(a**2)
        wavelength = wavelength_nm * 1e-9
        energy = h * c / wavelength
        photon_radiance = Hb / energy
        return photon_radiance * Omega * A_rx * Bf * Delta_t
    
    Hb = Hb_day if 'day' in condition_type else Hb_night
    return Nbg(Hb, Omega, Bf, Delta_t, a, wavelength_nm)


def run_qkd_simulation(condition_name, loss_file, num_pairs=500000):
    """
    Run complete QKD simulation for a given condition
    
    Args:
        condition_name: Name of condition (e.g., 'day1_72km')
        loss_file: Loss data filename
        num_pairs: Number of photon pairs to emit
    
    Returns:
        dict: Complete simulation results (or error dict if failed)
    """
    try:
        # Load loss data
        loss_array = elliptic_beam_model().load_data(f"loss_data/{loss_file}")
        
        # Calculate error probability
        error_prob = calculate_error_prob(condition_name)
    except Exception as e:
        return {
            'condition': condition_name,
            'secure': False,
            'error': True,
            'error_stage': 'initialization',
            'error_message': str(e),
            'message': f'Initialization failed: {str(e)}'
        }
    
    # Configure simulation
    config = {
        'num_pairs': num_pairs,
        'avg_pair_rate_hz': 100000,
        'source_fidelity_sq': 1.00,
        'node_distance_a': 72,
        'node_distance_b': 72,
        'loss_array': loss_array,
        'error_prob': error_prob,
        'jitter_components': [0.4, 0.5, 0.3]
    }
    
    # Reset logger
    logger = reset_logger()
    
    # === KEY DISTRIBUTION ===
    try:
        alice_collector, bob_collector, alice, bob = run_experiment(config)
    except Exception as e:
        return {
            'condition': condition_name,
            'secure': False,
            'error': True,
            'error_stage': 'key_distribution',
            'error_message': str(e),
            'message': f'Key distribution failed: {str(e)}'
        }
    
    # === COINCIDENCE DETECTION ===
    try:
        with open('qkd_raw_data/metadata.json') as f:
            data = json.load(f)
        
        is_positive = data['expected_delay_diff_ns'] > 0
        alice_c = CoincidenceDetection(alice, data_collector=alice_collector, 
                                       port=alice.get_conn_port(bob.ID),
                                       name="alice_c", first=is_positive)
        bob_c = CoincidenceDetection(bob, data_collector=bob_collector, 
                                     port=bob.get_conn_port(alice.ID),
                                     name="bob_c", first=not(is_positive))
        alice_c.start()
        bob_c.start()
        ns.sim_run()
        alice_c.stop()
        bob_c.stop()
    except Exception as e:
        return {
            'condition': condition_name,
            'secure': False,
            'error': True,
            'error_stage': 'coincidence_detection',
            'error_message': str(e),
            'message': f'Coincidence detection failed: {str(e)}'
        }
    
    # Check security
    try:
        with open('qkd_raw_data/metadata.json') as f:
            data = json.load(f)
        
        qber = data['qber']
        is_secure = data['is_secure']
    except Exception as e:
        return {
            'condition': condition_name,
            'secure': False,
            'error': True,
            'error_stage': 'qber_check',
            'error_message': str(e),
            'message': f'QBER check failed: {str(e)}'
        }
    
    if not is_secure:
        return {
            'condition': condition_name,
            'secure': False,
            'error': False,
            'qber': qber,
            'message': f'Channel not secure - QBER: {qber:.4f} > 0.11'
        }
    
    # === ERROR CORRECTION ===
    try:
        alice_r = CascadeServer(alice, port=alice.get_conn_port(bob.ID),
                               alice_key_file="alice.txt", name="alice_r")
        bob_r = CascadeClient(bob, port=bob.get_conn_port(alice.ID), 
                             bob_key_file="bob.txt",
                             algorithm_name="original", 
                             estimated_bit_error_rate=qber, 
                             name="bob_r")
        alice_r.start()
        bob_r.start()
        ns.sim_run()
        alice_r.stop()
        bob_r.stop()
        
        # Load reconciliation stats
        with open("reconciliation_stats.json", "r") as f:
            recon_stats = json.load(f)
        
        key_size = recon_stats["key_size"]
        reply_parity_bits = recon_stats["reply_parity_bits"]
    except Exception as e:
        return {
            'condition': condition_name,
            'secure': False,
            'error': True,
            'error_stage': 'error_correction',
            'error_message': str(e),
            'qber': qber,
            'message': f'Error correction failed: {str(e)}'
        }
    
    logger.log_event(
        source='system',
        sim_time=ns.sim_time(),
        event_type='reconciliation_complete',
        data={
            'reconciled_key_size': key_size,
            'reply_parity_bits': reply_parity_bits
        },
        message=f"Reconciled key: {key_size} bits"
    )
    
    # === PRIVACY AMPLIFICATION ===
    try:
        alice_pa = PrivacyAmplificationServer(
            alice, 
            port=alice.get_conn_port(bob.ID),
            alice_key_file="alice_recon.txt",
            output_file="alice_amplified.txt",
            name="alice_pa"
        )
        
        bob_pa = PrivacyAmplificationClient(
            bob,
            port=bob.get_conn_port(alice.ID),
            bob_key_file="bob_recon.txt", 
            output_file="bob_amplified.txt",
            name="bob_pa"
        )
        
        alice_pa.start()
        bob_pa.start()
        ns.sim_run()
        alice_pa.stop()
        bob_pa.stop()
        
        # Read final keys
        alice_final = read_key('alice_amplified.txt')
        bob_final = read_key('bob_amplified.txt')
    except Exception as e:
        return {
            'condition': condition_name,
            'secure': False,
            'error': True,
            'error_stage': 'privacy_amplification',
            'error_message': str(e),
            'qber': qber,
            'reconciled_key_bits': key_size,
            'message': f'Privacy amplification failed: {str(e)}'
        }
    
    logger.log_event(
        source='system',
        sim_time=ns.sim_time(),
        event_type='privacy_amplification_complete',
        data={
            'final_key_length': len(alice_final)
        },
        message=f"Privacy amplification complete! Final key: {len(alice_final)} bits"
    )
    
    # === COLLECT ALL RESULTS ===
    try:
        # Get all log events
        all_events = logger.events
        
        # Parse key events - using ACTUAL event type names from the logger
        key_dist_events = [e for e in all_events if e['event_type'] == 'completion']
        coincidence_events = [e for e in all_events if e['event_type'] == 'coincidence']
        basis_recon_events = [e for e in all_events if e['event_type'] == 'basis_matching']
        qber_events = [e for e in all_events if e['event_type'] == 'qber']
        recon_events = [e for e in all_events if e['event_type'] == 'reconciliation_complete']
        pa_events = [e for e in all_events if e['event_type'] == 'privacy_amplification_complete']
        
        # Check if we have all required events
        if not key_dist_events:
            raise ValueError("Missing 'completion' event (key distribution)")
        if not basis_recon_events:
            raise ValueError("Missing 'basis_matching' event")
        if not recon_events:
            raise ValueError("Missing 'reconciliation_complete' event")
        if not pa_events:
            raise ValueError("Missing 'privacy_amplification_complete' event")
        
        key_dist_event = key_dist_events[0]
        
        # Extract completion times for each stage
        # Sifted key is right after basis reconciliation (before error estimation sacrifice)
        time_after_sifting_ns = basis_recon_events[0]['sim_time_ns']
        time_after_reconciliation_ns = recon_events[0]['sim_time_ns']
        time_after_pa_ns = pa_events[0]['sim_time_ns']
        
        # Sifted key length is stored in 'matching_basis' field
        sifted_key_length = basis_recon_events[0]['data']['matching_basis']
        
        results = {
            'condition': condition_name,
            'secure': True,
            'error': False,
            'config': config,
            'emitted_pairs': num_pairs,
            'alice_detections': key_dist_event['data']['alice_detections'],
            'bob_detections': key_dist_event['data']['bob_detections'],
            'coincidences': coincidence_events[0]['data']['total_coincidences'],
            'sifted_key_bits': sifted_key_length,
            'qber': qber,
            'reconciled_key_bits': key_size,
            'final_key_bits': len(alice_final),
            'reconciliation_stats': recon_stats,
            'time_after_sifting_ns': time_after_sifting_ns,
            'time_after_reconciliation_ns': time_after_reconciliation_ns,
            'time_after_pa_ns': time_after_pa_ns,
            'time_after_sifting_s': time_after_sifting_ns / 1e9,
            'time_after_reconciliation_s': time_after_reconciliation_ns / 1e9,
            'time_after_pa_s': time_after_pa_ns / 1e9,
            'total_sim_time_ns': ns.sim_time(),
            'total_sim_time_s': ns.sim_time() / 1e9,
            'all_log_events': all_events
        }
        
        # Calculate rates using stage-specific completion times
        results['sifted_rate_bps'] = results['sifted_key_bits'] / results['time_after_sifting_s']
        results['sifted_rate_kbps'] = results['sifted_rate_bps'] / 1000
        
        results['reconciled_rate_bps'] = results['reconciled_key_bits'] / results['time_after_reconciliation_s']
        results['reconciled_rate_kbps'] = results['reconciled_rate_bps'] / 1000
        
        results['final_rate_bps'] = results['final_key_bits'] / results['time_after_pa_s']
        results['final_rate_kbps'] = results['final_rate_bps'] / 1000
        
        return results
        
    except Exception as e:
        return {
            'condition': condition_name,
            'secure': False,
            'error': True,
            'error_stage': 'result_collection',
            'error_message': str(e),
            'qber': qber,
            'reconciled_key_bits': key_size,
            'final_key_bits': len(alice_final) if 'alice_final' in locals() else None,
            'message': f'Result collection failed: {str(e)}'
        }


def run_multiple_trials(condition_name, loss_file, num_pairs=500000, num_trials=5):
    """
    Run multiple trials for a condition and collect statistics
    
    Args:
        condition_name: Name of condition
        loss_file: Loss data filename
        num_pairs: Number of photon pairs per trial
        num_trials: Number of trials to run
    
    Returns:
        dict: Aggregated results with statistics
    """
    all_results = []
    
    for trial in range(num_trials):
        print(f"  Trial {trial + 1}/{num_trials}...", end=' ', flush=True)
        
        try:
            result = run_qkd_simulation(condition_name, loss_file, num_pairs)
            result['trial_number'] = trial + 1
            all_results.append(result)
            
            if result.get('error', False):
                print(f"ERROR - {result['error_stage']}: {result['error_message']}")
            elif result['secure']:
                print(f"OK - Final key: {result['final_key_bits']} bits, Rate: {result['final_rate_kbps']:.3f} kbps")
            else:
                print(f"FAILED - {result['message']}")
        except Exception as e:
            # Catch any unexpected exceptions
            print(f"UNEXPECTED ERROR - {str(e)}")
            all_results.append({
                'condition': condition_name,
                'trial_number': trial + 1,
                'secure': False,
                'error': True,
                'error_stage': 'unexpected',
                'error_message': str(e),
                'message': f'Unexpected error: {str(e)}'
            })
    
    # Calculate statistics for secure runs only
    secure_results = [r for r in all_results if r.get('secure', False)]
    
    if len(secure_results) == 0:
        return {
            'condition': condition_name,
            'num_trials': num_trials,
            'num_secure': 0,
            'all_trials': all_results
        }
    
    # Aggregate metrics
    # Extract CASCADE efficiency with defensive handling
    cascade_efficiencies = []
    for r in secure_results:
        if 'reconciliation_stats' in r and r['reconciliation_stats'] is not None:
            eff = r['reconciliation_stats'].get('efficiency')
            if eff is not None:
                cascade_efficiencies.append(eff)
    
    aggregated = {
        'condition': condition_name,
        'loss_file': loss_file,
        'num_pairs': num_pairs,
        'num_trials': num_trials,
        'num_secure': len(secure_results),
        'avg_alice_detections': np.mean([r['alice_detections'] for r in secure_results]),
        'avg_bob_detections': np.mean([r['bob_detections'] for r in secure_results]),
        'avg_coincidences': np.mean([r['coincidences'] for r in secure_results]),
        'avg_sifted_key_bits': np.mean([r['sifted_key_bits'] for r in secure_results]),
        'avg_qber': np.mean([r['qber'] for r in secure_results]),
        'avg_reconciled_key_bits': np.mean([r['reconciled_key_bits'] for r in secure_results]),
        'avg_final_key_bits': np.mean([r['final_key_bits'] for r in secure_results]),
        'avg_sifted_rate_kbps': np.mean([r['sifted_rate_kbps'] for r in secure_results]),
        'avg_reconciled_rate_kbps': np.mean([r['reconciled_rate_kbps'] for r in secure_results]),
        'avg_final_rate_kbps': np.mean([r['final_rate_kbps'] for r in secure_results]),
        'avg_cascade_efficiency': np.mean(cascade_efficiencies) if cascade_efficiencies else None,
        'std_final_key_bits': np.std([r['final_key_bits'] for r in secure_results]),
        'std_sifted_rate_kbps': np.std([r['sifted_rate_kbps'] for r in secure_results]),
        'std_reconciled_rate_kbps': np.std([r['reconciled_rate_kbps'] for r in secure_results]),
        'std_final_rate_kbps': np.std([r['final_rate_kbps'] for r in secure_results]),
        'all_trials': all_results
    }
    
    return aggregated
