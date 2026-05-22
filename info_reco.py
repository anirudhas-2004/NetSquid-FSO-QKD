# Filename: info_reco.py
# Created: 15-10-2025
# Description: NetSquid adapter for cascade-python
# uses: https://github.com/brunorijsman/cascade-python

from netsquid.protocols import NodeProtocol
from netsquid.components.component import Message
import netsquid as ns
import copy
import heapq
import math
import time
import json
import logging
from cascade.block import Block
from cascade.algorithm import get_algorithm_by_name
from cascade.shuffle import Shuffle
from cascade.stats import Stats
from cascade.key import Key

# Configure logging
logging.basicConfig(
    filename='reconciliation.log',
    level=logging.INFO,
    format='%(message)s',  # Simplified format since we'll include sim time in messages
    filemode='w'  # Overwrite log file each run
)

class CascadeServer(NodeProtocol):
    def __init__(self, node, port, alice_key_file, output_key_file="alice_recon.txt", name=None):
        super().__init__(node=node, name=name or "CascadeServer")
        self.port = port
        
        # Load Alice's correct key from file
        self._correct_key = self._load_key_from_file(alice_key_file)
        self._output_key_file = output_key_file

    @staticmethod
    def _load_key_from_file(filename):
        """Load a key from a text file containing a binary string."""
        with open(filename, 'r') as f:
            key_string = f.read().strip()
        
        # Create a Key object
        key = Key()
        key._size = len(key_string)
        key._bits = {}
        
        for i, bit_char in enumerate(key_string):
            key._bits[i] = int(bit_char)
        
        return key

    def _save_key_to_file(self, key, filename):
        """Save a key to a text file as a binary string."""
        key_string = ""
        for i in range(key.get_size()):
            key_string += str(key.get_bit(i))
        
        with open(filename, 'w') as f:
            f.write(key_string)

    def run(self):
        """Main server loop - wait for parity requests and respond."""
        logging.info(f"[t={ns.sim_time():.0f}ns] SERVER: Starting CASCADE server protocol")
        logging.info(f"[t={ns.sim_time():.0f}ns] SERVER: Key size = {self._correct_key.get_size()} bits")
        
        port_ready = self.await_port_input(self.port)
        message_count = 0
        
        while True:
            yield port_ready
            
            classical_message = self.port.rx_input(header="recon")
            
            if classical_message:
                message_content = classical_message.items
                
                # Check if it's the end message
                if len(message_content) == 1 and message_content[0] == "end_reconciliation":
                    logging.info(f"[t={ns.sim_time():.0f}ns] SERVER: Received end_reconciliation signal")
                    logging.info(f"[t={ns.sim_time():.0f}ns] SERVER: Total parity requests processed: {message_count}")
                    self._end_reconcile()
                    break
                
                # Otherwise it's a parity request - items contains the list of blocks
                ask_parity_blocks = message_content
                message_count += 1
                logging.info(f"[t={ns.sim_time():.0f}ns] SERVER: Received parity request #{message_count} with {len(ask_parity_blocks)} blocks")
                
                correct_parities = self.ask_parities(ask_parity_blocks)
                self.port.tx_output(Message(correct_parities, header="recon"))
                logging.info(f"[t={ns.sim_time():.0f}ns] SERVER: Sent parity response #{message_count}")
    
    def ask_parities(self, blocks):
        """Calculate parities for the requested blocks."""
        parities = []
        for block in blocks:
            shuffle = block.get_shuffle()
            start_index = block.get_start_index()
            end_index = block.get_end_index()
            parity = shuffle.calculate_parity(self._correct_key, start_index, end_index)
            parities.append(parity)
        return parities
    
    def _end_reconcile(self):
        """Save Alice's key after reconciliation completes."""
        logging.info(f"[t={ns.sim_time():.0f}ns] SERVER: Saving reconciled key to {self._output_key_file}")
        self._save_key_to_file(self._correct_key, self._output_key_file)
        logging.info(f"[t={ns.sim_time():.0f}ns] SERVER: Reconciliation complete")


class CascadeClient(NodeProtocol):
    """
    NetSquid-compatible Cascade protocol client (Bob's side).
    Performs information reconciliation with the server (Alice's side).
    """
    
    def __init__(self, node, port, algorithm_name, bob_key_file, estimated_bit_error_rate, 
                 output_key_file="bob_recon.txt", stats_file="reconciliation_stats.json", name=None):
        super().__init__(node=node, name=name or "CascadeClient")
        self.port = port
        
        # Load Bob's noisy key from file
        self._noisy_key = self._load_key_from_file(bob_key_file)
        self._output_key_file = output_key_file
        self._stats_file = stats_file
        
        # Cascade parameters
        self._algorithm = get_algorithm_by_name(algorithm_name)
        assert self._algorithm is not None
        self._estimated_bit_error_rate = estimated_bit_error_rate
        self._reconciled_key = None
        
        # Map key indexes to blocks
        self._key_index_to_blocks = {}
        
        # Statistics
        self.stats = Stats()
        
        # Priority queue for blocks to correct (min-heap by block size)
        self._pending_try_correct = []
        
        # Queue for parity questions
        self._pending_ask_correct_parity = []

    @staticmethod
    def _load_key_from_file(filename):
        """Load a key from a text file containing a binary string."""
        with open(filename, 'r') as f:
            key_string = f.read().strip()
        
        # Create a Key object
        key = Key()
        key._size = len(key_string)
        key._bits = {}
        
        for i, bit_char in enumerate(key_string):
            key._bits[i] = int(bit_char)
        
        return key

    def _save_key_to_file(self, key, filename):
        """Save a key to a text file as a binary string."""
        key_string = ""
        for i in range(key.get_size()):
            key_string += str(key.get_bit(i))
        
        with open(filename, 'w') as f:
            f.write(key_string)

    def _save_stats_to_json(self, filename):
        """Save statistics to a JSON file."""
        stats_dict = {
            "elapsed_process_time": self.stats.elapsed_process_time,
            "elapsed_real_time": self.stats.elapsed_real_time,
            "normal_iterations": self.stats.normal_iterations,
            "biconf_iterations": self.stats.biconf_iterations,
            "ask_parity_messages": self.stats.ask_parity_messages,
            "ask_parity_blocks": self.stats.ask_parity_blocks,
            "ask_parity_bits": self.stats.ask_parity_bits,
            "reply_parity_bits": self.stats.reply_parity_bits,
            "reconciliation_bits": self.stats.reconciliation_bits,
            "reconciliation_bits_per_key_bit": self.stats.reconciliation_bits_per_key_bit,
            "efficiency": self.stats.efficiency,
            "infer_parity_blocks": self.stats.infer_parity_blocks,
            "key_size": self._noisy_key.get_size(),
            "estimated_bit_error_rate": self._estimated_bit_error_rate
        }
        
        with open(filename, 'w') as f:
            json.dump(stats_dict, f, indent=4)

    def run(self):
        """
        Main protocol execution - runs the complete Cascade reconciliation.
        """
        logging.info(f"[t={ns.sim_time():.0f}ns] CLIENT: Starting CASCADE reconciliation protocol")
        logging.info(f"[t={ns.sim_time():.0f}ns] CLIENT: Algorithm: {self._algorithm.name}")
        logging.info(f"[t={ns.sim_time():.0f}ns] CLIENT: Key size: {self._noisy_key.get_size()} bits")
        logging.info(f"[t={ns.sim_time():.0f}ns] CLIENT: Estimated BER: {self._estimated_bit_error_rate}")
        
        # Start measuring time
        start_process_time = time.process_time()
        start_real_time = time.perf_counter()
        
        # Make a deep copy of the key
        self._reconciled_key = copy.deepcopy(self._noisy_key)
        
        # CRITICAL FIX: Add initial yield to ensure protocol timing is correct
        # This allows Alice's server to be ready before we start sending messages
        #yield self.await_timer(0)
        
        # Do normal Cascade iterations
        logging.info(f"[t={ns.sim_time():.0f}ns] CLIENT: Starting normal CASCADE iterations")
        yield from self._all_normal_cascade_iterations()
        logging.info(f"[t={ns.sim_time():.0f}ns] CLIENT: Completed {self.stats.normal_iterations} normal iterations")
        
        # Do BICONF iterations
        logging.info(f"[t={ns.sim_time():.0f}ns] CLIENT: Starting BICONF iterations")
        yield from self._all_biconf_iterations()
        logging.info(f"[t={ns.sim_time():.0f}ns] CLIENT: Completed {self.stats.biconf_iterations} BICONF iterations")
        
        # Inform Alice we're done
        logging.info(f"[t={ns.sim_time():.0f}ns] CLIENT: Sending end_reconciliation signal")
        self.port.tx_output(Message("end_reconciliation", header="recon"))
        
        # Compute statistics
        self.stats.elapsed_process_time = time.process_time() - start_process_time
        self.stats.elapsed_real_time = time.perf_counter() - start_real_time
        self.stats.efficiency = self._compute_efficiency(self.stats.ask_parity_blocks)
        self.stats.reconciliation_bits = (self.stats.ask_parity_bits + 
                                         self.stats.reply_parity_bits)
        key_size = self._noisy_key.get_size()
        self.stats.reconciliation_bits_per_key_bit = (float(self.stats.reconciliation_bits) / 
                                                      float(key_size))
        
        logging.info(f"[t={ns.sim_time():.0f}ns] CLIENT: === RECONCILIATION STATISTICS ===")
        logging.info(f"[t={ns.sim_time():.0f}ns] CLIENT: Ask parity messages: {self.stats.ask_parity_messages}")
        logging.info(f"[t={ns.sim_time():.0f}ns] CLIENT: Ask parity blocks: {self.stats.ask_parity_blocks}")
        logging.info(f"[t={ns.sim_time():.0f}ns] CLIENT: Inferred parity blocks: {self.stats.infer_parity_blocks}")
        logging.info(f"[t={ns.sim_time():.0f}ns] CLIENT: Reconciliation bits: {self.stats.reconciliation_bits}")
        logging.info(f"[t={ns.sim_time():.0f}ns] CLIENT: Bits per key bit: {self.stats.reconciliation_bits_per_key_bit:.4f}")
        logging.info(f"[t={ns.sim_time():.0f}ns] CLIENT: Efficiency: {self.stats.efficiency:.4f}" if self.stats.efficiency else f"[t={ns.sim_time():.0f}ns] CLIENT: Efficiency: N/A")
        
        # Save the reconciled key to file
        logging.info(f"[t={ns.sim_time():.0f}ns] CLIENT: Saving reconciled key to {self._output_key_file}")
        self._save_key_to_file(self._reconciled_key, self._output_key_file)
        
        # Save statistics to JSON file
        logging.info(f"[t={ns.sim_time():.0f}ns] CLIENT: Saving statistics to {self._stats_file}")
        self._save_stats_to_json(self._stats_file)
        
        logging.info(f"[t={ns.sim_time():.0f}ns] CLIENT: Reconciliation protocol completed successfully")
        
        return self._reconciled_key

    def _all_normal_cascade_iterations(self):
        """Execute all normal Cascade iterations."""
        for iteration_nr in range(1, self._algorithm.cascade_iterations + 1):
            yield from self._one_normal_cascade_iteration(iteration_nr)

    def _one_normal_cascade_iteration(self, iteration_nr):
        """Execute one normal Cascade iteration."""
        self.stats.normal_iterations += 1
        
        # Determine block size
        block_size = self._algorithm.block_size_function(
            self._estimated_bit_error_rate,
            self._reconciled_key.get_size(),
            iteration_nr
        )
        
        logging.info(f"[t={ns.sim_time():.0f}ns] CLIENT: Normal iteration {iteration_nr}, block_size={block_size}")
        
        # Create shuffle
        if iteration_nr == 1:
            shuffle = Shuffle(self._reconciled_key.get_size(), Shuffle.SHUFFLE_KEEP_SAME)
        else:
            shuffle = Shuffle(self._reconciled_key.get_size(), Shuffle.SHUFFLE_RANDOM)
        
        # Create covering blocks
        blocks = Block.create_covering_blocks(self._reconciled_key, shuffle, block_size)
        logging.info(f"[t={ns.sim_time():.0f}ns] CLIENT: Created {len(blocks)} covering blocks")
        
        # Schedule each block for parity check
        for block in blocks:
            self._register_block_key_indexes(block)
            self._schedule_ask_correct_parity(block, False)
        
        # Service all pending work
        yield from self._service_all_pending_work(True)

    def _all_biconf_iterations(self):
        """Execute all BICONF iterations."""
        if not self._algorithm.biconf_iterations:
            return
        
        # Clear key index map if not cascading during BICONF
        if not self._algorithm.biconf_cascade:
            self._key_index_to_blocks = {}
        
        # Do required number of BICONF iterations
        iterations_to_go = self._algorithm.biconf_iterations
        while iterations_to_go > 0:
            errors_corrected = yield from self._one_biconf_iteration()
            if self._algorithm.biconf_error_free_streak and errors_corrected > 0:
                iterations_to_go = self._algorithm.biconf_iterations
            else:
                iterations_to_go -= 1

    def _one_biconf_iteration(self):
        """Execute one BICONF iteration."""
        self.stats.biconf_iterations += 1
        cascade = self._algorithm.biconf_cascade
        
        logging.info(f"[t={ns.sim_time():.0f}ns] CLIENT: BICONF iteration {self.stats.biconf_iterations}")
        
        # Randomly select half of the key bits
        key_size = self._reconciled_key.get_size()
        shuffle = Shuffle(key_size, Shuffle.SHUFFLE_RANDOM)
        mid_index = key_size // 2
        chosen_block = Block(self._reconciled_key, shuffle, 0, mid_index, None)
        
        if cascade:
            self._register_block_key_indexes(chosen_block)
        
        # Ask for parity
        self._schedule_ask_correct_parity(chosen_block, False)
        
        # Check complement if required
        if self._algorithm.biconf_correct_complement:
            complement_block = Block(self._reconciled_key, shuffle, mid_index, key_size, None)
            if cascade:
                self._register_block_key_indexes(complement_block)
            self._schedule_ask_correct_parity(complement_block, False)
        
        # Service all pending work
        errors_corrected = yield from self._service_all_pending_work(cascade)
        logging.info(f"[t={ns.sim_time():.0f}ns] CLIENT: BICONF iteration corrected {errors_corrected} errors")
        return errors_corrected

    def _service_all_pending_work(self, cascade):
        """Service all pending corrections and parity requests."""
        errors_corrected = 0
        
        while self._have_pending_try_correct() or self._have_pending_ask_correct_parity():
            # Try to correct pending blocks
            errors_corrected += self._service_pending_try_correct(cascade)
            
            # Ask Alice for parities (async operation)
            if self._have_pending_ask_correct_parity():
                yield from self._service_pending_ask_correct_parity()
        
        return errors_corrected

    def _service_pending_ask_correct_parity(self):
        """Send parity requests to Alice and wait for response."""
        if not self._pending_ask_correct_parity:
            return
        
        # Prepare the question
        ask_parity_blocks = []
        for entry in self._pending_ask_correct_parity:
            (block, _correct_right_sibling) = entry
            ask_parity_blocks.append(block)
            self.stats.ask_parity_bits += self._bits_in_block_ask_parity(block)
        
        # Send to Alice
        self.stats.ask_parity_messages += 1
        self.stats.ask_parity_blocks += len(ask_parity_blocks)
        
        logging.info(f"[t={ns.sim_time():.0f}ns] CLIENT: Sending parity request with {len(ask_parity_blocks)} blocks")
        self.port.tx_output(Message(ask_parity_blocks, header="recon"))
        
        # Wait for Alice's response
        port_ready = self.await_port_input(self.port)
        yield port_ready
        
        classical_message = self.port.rx_input(header="recon")
        
        if classical_message:
            correct_parities = classical_message.items
            logging.info(f"[t={ns.sim_time():.0f}ns] CLIENT: Received parity response with {len(correct_parities)} parities")
            
            # Process the response
            for (correct_parity, entry) in zip(correct_parities, self._pending_ask_correct_parity):
                self.stats.reply_parity_bits += 1
                (block, correct_right_sibling) = entry
                block.set_correct_parity(correct_parity)
                self._schedule_try_correct(block, correct_right_sibling)
            
            # Clear pending questions
            self._pending_ask_correct_parity = []

    def _service_pending_try_correct(self, cascade):
        """Attempt to correct all pending blocks."""
        errors_corrected = 0
        while self._pending_try_correct:
            (_, entry) = heapq.heappop(self._pending_try_correct)
            (block, correct_right_sibling) = entry
            errors_corrected += self._try_correct(block, correct_right_sibling, cascade)
        return errors_corrected

    def _try_correct(self, block, correct_right_sibling, cascade):
        """Try to correct errors in a block."""
        # Check if we know the correct parity
        if not self._correct_parity_is_known_or_can_be_inferred(block):
            self._schedule_ask_correct_parity(block, correct_right_sibling)
            return 0
        
        # Even number of errors - maybe correct right sibling
        if block.get_error_parity() == Block.ERRORS_EVEN:
            if correct_right_sibling:
                return self._try_correct_right_sibling_block(block, cascade)
            return 0
        
        # Single bit - correct it!
        if block.get_size() == 1:
            self._flip_key_bit_corresponding_to_single_bit_block(block, cascade)
            return 1
        
        # Recurse to left sub-block
        left_sub_block = block.get_left_sub_block()
        if left_sub_block is None:
            left_sub_block = block.create_left_sub_block()
            self._register_block_key_indexes(left_sub_block)
        return self._try_correct(left_sub_block, True, cascade)

    def _try_correct_right_sibling_block(self, block, cascade):
        """Try to correct the right sibling block."""
        parent_block = block.get_parent_block()
        right_sibling_block = parent_block.get_right_sub_block()
        if right_sibling_block is None:
            right_sibling_block = parent_block.create_right_sub_block()
            self._register_block_key_indexes(right_sibling_block)
        return self._try_correct(right_sibling_block, False, cascade)

    def _flip_key_bit_corresponding_to_single_bit_block(self, block, cascade):
        """Flip the key bit corresponding to a single-bit block."""
        flipped_shuffle_index = block.get_start_index()
        block.flip_bit(flipped_shuffle_index)
        
        flipped_key_index = block.get_key_index(flipped_shuffle_index)
        logging.info(f"[t={ns.sim_time():.0f}ns] CLIENT: Flipped bit at key index {flipped_key_index}")
        
        for affected_block in self._get_blocks_containing_key_index(flipped_key_index):
            affected_block.flip_parity()
            
            if cascade and affected_block.get_error_parity() != Block.ERRORS_EVEN:
                if self._algorithm.sub_block_reuse or affected_block.is_top_block():
                    self._schedule_try_correct(affected_block, False)

    # Helper methods
    def _register_block_key_indexes(self, block):
        """Register block in the key index map."""
        for key_index in block.get_key_indexes():
            if key_index in self._key_index_to_blocks:
                self._key_index_to_blocks[key_index].append(block)
            else:
                self._key_index_to_blocks[key_index] = [block]

    def _get_blocks_containing_key_index(self, key_index):
        """Get all blocks containing a specific key index."""
        return self._key_index_to_blocks.get(key_index, [])

    def _correct_parity_is_known_or_can_be_inferred(self, block):
        """Check if correct parity is known or can be inferred."""
        if block.get_correct_parity() is not None:
            return True
        
        parent_block = block.get_parent_block()
        if parent_block is None:
            return False
        
        if parent_block.get_left_sub_block() == block:
            sibling_block = parent_block.get_right_sub_block()
        else:
            sibling_block = parent_block.get_left_sub_block()
        
        if sibling_block is None:
            return False
        
        correct_parent_parity = parent_block.get_correct_parity()
        if correct_parent_parity is None:
            return False
        
        correct_sibling_parity = sibling_block.get_correct_parity()
        if correct_sibling_parity is None:
            return False
        
        # Infer the parity
        if correct_parent_parity == 1:
            correct_block_parity = 1 - correct_sibling_parity
        else:
            correct_block_parity = correct_sibling_parity
        
        block.set_correct_parity(correct_block_parity)
        self.stats.infer_parity_blocks += 1
        return True

    def _schedule_ask_correct_parity(self, block, correct_right_sibling):
        """Schedule a parity request."""
        entry = (block, correct_right_sibling)
        self._pending_ask_correct_parity.append(entry)

    def _schedule_try_correct(self, block, correct_right_sibling):
        """Schedule a correction attempt."""
        entry = (block, correct_right_sibling)
        heapq.heappush(self._pending_try_correct, (block.get_size(), entry))

    def _have_pending_ask_correct_parity(self):
        """Check if there are pending parity requests."""
        return self._pending_ask_correct_parity != []

    def _have_pending_try_correct(self):
        """Check if there are pending correction attempts."""
        return self._pending_try_correct != []

    @staticmethod
    def _bits_in_int(int_value):
        """Calculate bits needed to represent an integer."""
        bits = 0
        while int_value != 0:
            bits += 1
            int_value //= 2
        return bits if bits > 0 else 1

    @staticmethod
    def _bits_in_block_ask_parity(block):
        """Calculate bits needed to ask for parity of a block."""
        shuffle_identifier = block.get_shuffle().get_identifier()
        shuffle_start_index = block.get_start_index()
        shuffle_end_index = block.get_end_index()
        return (CascadeClient._bits_in_int(shuffle_identifier) +
                CascadeClient._bits_in_int(shuffle_start_index) +
                CascadeClient._bits_in_int(shuffle_end_index))

    def _compute_efficiency(self, reconciliation_bits):
        """Compute Shannon efficiency."""
        eps = self._estimated_bit_error_rate
        try:
            shannon_efficiency = -eps * math.log2(eps) - (1 - eps) * math.log2(1 - eps)
            key_size = self._noisy_key.get_size()
            efficiency = reconciliation_bits / (key_size * shannon_efficiency)
        except (ValueError, ZeroDivisionError):
            efficiency = None
        return efficiency

    def get_reconciled_key(self):
        """Get the reconciled key."""
        return self._reconciled_key

    def get_noisy_key(self):
        """Get the original noisy key."""
        return self._noisy_key
