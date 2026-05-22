# Filename: qkd_logger.py
# Created: 05-11-2025
# Description: Enables all sub-processes to create and modify global variables
# Useful when running the whole simulation in a GUI: qkdgui.py

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

class QKDSimulationLogger:
    """
    Centralized logger for QKD simulation events.
    Captures events from Alice, Bob, and Source with timestamps.
    """
    
    def __init__(self, output_dir: str = "qkd_raw_data"):
        self.output_dir = output_dir
        self.events: List[Dict] = []
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def log_event(self, source: str, sim_time: float, event_type: str, 
                  data: Optional[Dict] = None, message: str = ""):
        """
        Log a simulation event.
        
        Args:
            source: Event source ('alice', 'bob', 'source', 'system')
            sim_time: Simulation time in nanoseconds
            event_type: Type of event ('detection', 'coincidence', 'qber', 'key_generation', etc.)
            data: Optional dictionary with event-specific data
            message: Human-readable message
        """
        event = {
            'run_id': self.run_id,
            'source': source,
            'sim_time_ns': sim_time,
            'event_type': event_type,
            'data': data or {},
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        self.events.append(event)
        
    def get_formatted_message(self, event: Dict) -> str:
        """Format event for display."""
        return f"[{event['source']:8s}] t={event['sim_time_ns']/1e9:.6f}s: {event['message']}"
    
    def get_recent_events(self, n: int = 10) -> List[str]:
        """Get last n events formatted for display."""
        return [self.get_formatted_message(e) for e in self.events[-n:]]
    
    def get_events_by_type(self, event_type: str) -> List[Dict]:
        """Get all events of a specific type."""
        return [e for e in self.events if e['event_type'] == event_type]
    
    def save_to_csv(self, filename: Optional[str] = None):
        """Save all events to CSV for analysis."""
        if filename is None:
            filename = os.path.join(self.output_dir, f"simulation_log_{self.run_id}.csv")
        
        # Flatten the data dictionary for CSV
        flattened_events = []
        for event in self.events:
            flat_event = {
                'run_id': event['run_id'],
                'source': event['source'],
                'sim_time_ns': event['sim_time_ns'],
                'event_type': event['event_type'],
                'message': event['message'],
                'timestamp': event['timestamp']
            }
            # Add data fields with prefix
            for key, value in event['data'].items():
                flat_event[f'data_{key}'] = value
            flattened_events.append(flat_event)
        
        df = pd.DataFrame(flattened_events)
        df.to_csv(filename, index=False)
        return filename
    
    def save_to_json(self, filename: Optional[str] = None):
        """Save all events to JSON."""
        if filename is None:
            filename = os.path.join(self.output_dir, f"simulation_log_{self.run_id}.json")
        
        with open(filename, 'w') as f:
            json.dump(self.events, f, indent=2)
        return filename
    
    def clear(self):
        """Clear all logged events."""
        self.events.clear()
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")


# Global logger instance
_logger_instance = None

def get_logger(output_dir: str = "qkd_raw_data") -> QKDSimulationLogger:
    """Get or create the global logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = QKDSimulationLogger(output_dir)
    return _logger_instance

def reset_logger(output_dir: str = "qkd_raw_data") -> QKDSimulationLogger:
    """Reset the global logger instance."""
    global _logger_instance
    _logger_instance = QKDSimulationLogger(output_dir)
    return _logger_instance
