# The MIT License (MIT)
# Copyright © 2023 Crazydevlegend
# Copyright © 2023 Rapiiidooo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import asyncio
import base64
import hashlib
import json
import os
import random
import tempfile
import threading
import traceback
import uuid
import numpy as np
from asyncio import AbstractEventLoop
from typing import Dict, Tuple, List
from pathlib import Path

import bittensor as bt
import time
import paramiko
import requests

import torch
from torch._C._te import Tensor  # type: ignore
import RSAEncryption as rsa
import concurrent.futures
from collections import defaultdict

import Validator.app_generator as ag
from compute import (
    SUSPECTED_EXPLOITERS_HOTKEYS,
    SUSPECTED_EXPLOITERS_COLDKEYS,
    __version_as_int__,
    validator_permit_stake,
    weights_rate_limit
)
from compute.axon import ComputeSubnetSubtensor
from compute.protocol import Allocate
from compute.pubsub import PubSubClient
from compute.utils.db import ComputeDb
from compute.utils.math import percent
from compute.utils.parser import ComputeArgPaser
from compute.utils.subtensor import is_registered, get_current_block, calculate_next_block_time
from compute.utils.version import try_update, get_local_version, version2number, get_remote_version
from compute.wandb.wandb import ComputeWandb
from neurons.Validator.calculate_pow_score import calc_score_pog
from neurons.Validator.database.allocate import update_miner_details, get_miner_details
from neurons.Validator.database.miner import select_miners, purge_miner_entries, update_miners
from neurons.Validator.health_check import perform_health_check
from neurons.Validator.pog import prng, adjust_matrix_size, compute_script_hash, execute_script_on_miner, get_random_seeds, load_yaml_config, parse_merkle_output, receive_responses, send_challenge_indices, send_script_and_request_hash, parse_benchmark_output, identify_gpu, send_seeds, verify_merkle_proof_row, get_remote_gpu_info, verify_responses, merkle_ok
from neurons.Validator.database.pog import get_pog_specs, retrieve_stats, update_pog_stats, write_stats, purge_pog_stats

class Validator:
    blocks_done: set = set()

    pow_requests: dict = {}
    pow_responses: dict = {}
    pow_benchmark: dict = {}
    new_pow_benchmark: dict = {}
    pow_benchmark_success: dict = {}

    queryable_for_specs: dict = {}
    finalized_specs_once: bool = False

    total_current_miners: int = 0

    scores: Tensor
    stats: dict

    validator_subnet_uid: int

    _queryable_uids: Dict[int, bt.AxonInfo]

    loop: AbstractEventLoop

    @property
    def wallet(self) -> bt.wallet: # type: ignore
        return self._wallet

    @property
    def subtensor(self) -> ComputeSubnetSubtensor:
        return self._subtensor


    @property
    def metagraph(self) -> bt.metagraph: # type: ignore
        return self._metagraph

    @property
    def queryable(self):
        return self._queryable_uids

    @property
    def queryable_uids(self):
        return [uid for uid in self._queryable_uids.keys()]

    @property
    def queryable_axons(self):
        return [axon for axon in self._queryable_uids.values()]

    @property
    def queryable_hotkeys(self):
        return [axon.hotkey for axon in self._queryable_uids.values()]

    @property
    def current_block(self):
        return get_current_block(subtensor=self.subtensor)

    @property
    def miners_items_to_set(self):
        return set((uid, hotkey) for uid, hotkey in self.miners.items()) if self.miners else None

    def __init__(self):
        # Step 1: Parse the bittensor and compute subnet config
        self.config = self.init_config()

        # Setup extra args
        self.blacklist_hotkeys = {hotkey for hotkey in self.config.blacklist_hotkeys}
        self.blacklist_coldkeys = {coldkey for coldkey in self.config.blacklist_coldkeys}
        self.whitelist_hotkeys = {hotkey for hotkey in self.config.whitelist_hotkeys}
        self.whitelist_coldkeys = {coldkey for coldkey in self.config.whitelist_coldkeys}
        self.exploiters_hotkeys = {hotkey for hotkey in SUSPECTED_EXPLOITERS_HOTKEYS} if self.config.blacklist_exploiters else {}
        self.exploiters_coldkeys = {coldkey for coldkey in SUSPECTED_EXPLOITERS_COLDKEYS} if self.config.blacklist_exploiters else {}

        # Set custom validator arguments
        self.validator_specs_batch_size = self.config.validator_specs_batch_size
        self.validator_challenge_batch_size = self.config.validator_challenge_batch_size
        self.validator_perform_hardware_query = self.config.validator_perform_hardware_query
        self.validator_whitelist_updated_threshold = self.config.validator_whitelist_updated_threshold

        # Set up logging with the provided configuration and directory.
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(f"Running validator for subnet: {self.config.netuid} on network: {self.config.subtensor.chain_endpoint} with config:")
        # Log the configuration for reference.
        bt.logging.info(self.config)

        # Step 2: Build Bittensor validator objects
        # These are core Bittensor classes to interact with the network.
        bt.logging.info("Setting up bittensor objects.")

        # The wallet holds the cryptographic key pairs for the validator.
        self._wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}")

        # The subtensor is our connection to the Bittensor blockchain.
        self._subtensor = ComputeSubnetSubtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")

        # The metagraph holds the state of the network, letting us know about other miners.
        self._metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        self.pubsub_client = PubSubClient(
            wallet=self.wallet,
            config=self.config,
            timeout=30.0,
            auto_refresh_interval=600  # 30 minutes
        )

        # Initialize the local db
        self.db = ComputeDb()
        self.miners: dict = select_miners(self.db)

        # Initialize wandb
        self.wandb = ComputeWandb(self.config, self.wallet, os.path.basename(__file__))

        # STEP 2B: Init Proof of GPU
        # Load configuration from YAML
        config_file = "config.yaml"
        self.config_data = load_yaml_config(config_file)

        # Bring everything else into memory on init, too
        self.gpu_performance  = self.config_data.get("gpu_performance", {})
        self.gpu_time_models  = self.config_data.get("gpu_time_models", {})
        self.merkle_proof     = self.config_data.get("merkle_proof", {})

        # ── server_settings ─────────────────────────────────────
        srv = self.config_data.get("server_settings", {})
        self.server_ip   = srv.get("server_ip",   "65.108.33.88")
        self.server_port = srv.get("server_port", "8000")
        self.server_url  = f"http://{self.server_ip}:{self.server_port}"

        self._last_cfg_pull    = 0.0
        self._cfg_pull_interval = srv.get("pull_interval",300)

        # immediately apply the disk‐based subnet_config
        self.refresh_config_from_server()
        self.load_subnet_config()

        cpu_cores = os.cpu_count() or 1
        configured_max_workers = self.config_data["merkle_proof"].get("max_workers", 32)
        safe_max_workers = min((cpu_cores + 4)*4, configured_max_workers)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=safe_max_workers)
        self.results = {}
        self.gpu_task = None  # Track the GPU task

        # Initialize allocated_hotkeys as an empty list
        self.allocated_hotkeys = []

        # Initialize penalized_hotkeys as an empty list
        self.penalized_hotkeys = []

        # Initialize penalized_hotkeys_checklist as an empty list
        self.penalized_hotkeys_checklist = []

        # Step 3: Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.uids: list = self.metagraph.uids.tolist()
        self.last_uids: list = self.uids.copy()
        self.init_scores()
        self.sync_status()

        self.last_updated_block = self.current_block - (self.current_block % 100)

        # Init the thread.
        self.lock = threading.Lock()
        self.threads: List[threading.Thread] = []

    @staticmethod
    def init_config():
        """
        This function is responsible for setting up and parsing command-line arguments.
        :return: config
        """
        parser = ComputeArgPaser(description="This script aims to help validators with the compute subnet.")
        config = parser.config

        # Step 3: Set up logging directory
        # Logging is crucial for monitoring and debugging purposes.
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey,
                config.netuid,
                "validator",
            )
        )
        # Ensure the logging directory exists.
        if not os.path.exists(config.full_path):
            os.makedirs(config.full_path, exist_ok=True)

        # Return the parsed config.
        return config

    def init_prometheus(self):
        """
        Register the prometheus information on metagraph.
        :return: bool
        """
        # extrinsic prometheus is removed at 8.2.1

        bt.logging.info("Extrinsic prometheus information on metagraph.")

        success = self._subtensor.serve_prometheus(
            wallet=self.wallet,
            port=bt.core.settings.DEFAULTS.axon.port,
            netuid=self.config.netuid
        )
        if success:
            bt.logging.success(
                prefix="Prometheus served",
                suffix=f"<blue>Current version: {get_local_version()}</blue>"  # Corrected keyword
            )
        else:
            bt.logging.error("Prometheus initialization failed")
        return success

    def init_local(self):
        bt.logging.info(f"🔄 Syncing metagraph with subtensor.")
        self._metagraph = self.subtensor.metagraph(self.config.netuid)
        self.uids = self.metagraph.uids.tolist()

    def init_scores(self):
        self.scores = torch.zeros(len(self.uids), dtype=torch.float32)
        # Set the weights of validators to zero.
        self.scores = self.scores * (self.metagraph.total_stake < 1.024e3)
        # Set the weight to zero for all nodes without assigned IP addresses.
        self.scores = self.scores * torch.Tensor(self.get_valid_tensors(metagraph=self.metagraph))
        bt.logging.info(f"🔢 Initialized scores : {self.scores.tolist()}")
        self.sync_scores()

    def refresh_config_from_server(self):
        """
        Every `_cfg_pull_interval` seconds, fetch the latest JSON config
        from your Streamlit/FastAPI endpoint and re‐apply *all* blocks.
        """
        now = time.time()
        if now - self._last_cfg_pull < self._cfg_pull_interval:
            return
        self._last_cfg_pull = now

        try:
            r = requests.get(f"{self.server_url}/config", timeout=5)
            if r.status_code != 200:
                bt.logging.warning(f"Could not fetch config: HTTP {r.status_code}")
                return

            new_cfg = r.json().get("config", {})
            if not isinstance(new_cfg, dict):
                bt.logging.warning("Remote config payload was not a dict")
                return

            # replace our in‐memory YAML dump
            self.config_data.update(new_cfg)

            # re‐load each section
            self.load_subnet_config()
            self.gpu_performance = new_cfg.get("gpu_performance", {})
            self.gpu_time_models = new_cfg.get("gpu_time_models", {})
            self.merkle_proof    = new_cfg.get("merkle_proof", {})

            bt.logging.info("🔄 Loaded updated config from server.")
        except Exception as e:
            bt.logging.warning(f"Error refreshing config: {e}")

    def load_subnet_config(self):
        subnet_config = self.config_data.get("subnet_config", {})

        # Scheduling constants
        self.blocks_per_epoch = subnet_config.get("blocks_per_epoch", 120)
        self.max_challenge_blocks = subnet_config.get("max_challenge_blocks", 10)
        self.rand_delay_blocks_max = subnet_config.get("rand_delay_blocks_max", 5)
        self.allow_fake_sybil_slot = subnet_config.get("allow_fake_sybil_slot", False)
        self.sybil_eligible_hotkeys = set(
            subnet_config.get("sybil_check_eligible_hotkeys") or []
        )

        # Emission control
        self.total_miner_emission = float(subnet_config.get("total_miner_emission", 0.0))
        self.gpu_weights = subnet_config.get("gpu_weights", {})

        bt.logging.debug(f"🔧 Loaded subnet config:")
        bt.logging.debug(f"  total_miner_emission = {self.total_miner_emission}")

    @staticmethod
    def pretty_print_dict_values(items: dict):
        for key, values in items.items():
            log = f"uid: {key}"

            for values_key, values_values in values.items():
                if values_key == "ss58_address":
                    values_values = values_values[:8] + (values_values[8:] and "...")
                try:
                    values_values = f"{float(values_values):.2f}"
                except Exception:
                    pass
                log += f" | {values_key}: {values_values}"

            bt.logging.trace(log)

    def update_allocation_wandb(self):
        hotkey_list = []
        # Instantiate the connection to the db
        cursor = self.db.get_cursor()
        try:
            # Retrieve all records from the allocation table
            cursor.execute("SELECT id, hotkey, details FROM allocation")
            rows = cursor.fetchall()
            for row in rows:
                id, hotkey, details = row
                hotkey_list.append(hotkey)
        except Exception as e:
            bt.logging.info(f"An error occurred while retrieving allocation details: {e}")
        finally:
            cursor.close()

        # Update wandb
        try:
            self.wandb.update_allocated_hotkeys(hotkey_list, self.penalized_hotkeys)
        except Exception as e:
            bt.logging.info(f"Error updating wandb : {e}")

    def sync_scores(self):
        # Fetch scoring stats
        self.stats = retrieve_stats(self.db)
        miner_details_all = get_miner_details(self.db)

        valid_validator_hotkeys = self.get_valid_validator_hotkeys()

        self.update_allocation_wandb()

        # Fetch allocated hotkeys and stats
        self.allocated_hotkeys = self.wandb.get_allocated_hotkeys(valid_validator_hotkeys, True)
        self.stats_allocated = self.wandb.get_stats_allocated(valid_validator_hotkeys, True)
        penalized_hotkeys = self.wandb.get_penalized_hotkeys_checklist(valid_validator_hotkeys, True)
        self._queryable_uids = self.get_queryable()

        # Calculate score
        for uid in self.uids:
            try:
                if uid not in self._queryable_uids:
                    hotkey = self.metagraph.axons[uid].hotkey
                    self.stats[uid] = {
                        "hotkey": hotkey,
                        "allocated": hotkey in self.allocated_hotkeys,
                        "own_score": True,
                        "score": 0,
                        "gpu_specs": None,
                        "reliability_score": 0.0
                        }
                    self.scores[uid] = 0

                    # Remove entry from PoG stats
                    cursor = self.db.get_cursor()
                    cursor.execute(
                        "DELETE FROM pog_stats WHERE hotkey = ?",
                        (hotkey,),
                    )
                    continue  # Skip further processing for this uid

                axon = self._queryable_uids[uid]
                hotkey = axon.hotkey

                if uid not in self.stats:
                    self.stats[uid] = {}

                self.stats[uid]["hotkey"] = hotkey

                # Mark whether this hotkey is in the allocated list
                self.stats[uid]["allocated"] = hotkey in self.allocated_hotkeys

                # Check GPU specs in our PoG DB
                gpu_specs = get_pog_specs(self.db, hotkey)

                # If found in our local database
                if gpu_specs is not None:
                    score = calc_score_pog(gpu_specs, hotkey, self.allocated_hotkeys, self.config_data)
                    self.stats[uid]["own_score"] = True  # or "yes" if you prefer a string
                else:
                    # If not found locally, try fallback from stats_allocated
                    if uid in self.stats_allocated:
                        if isinstance(self.stats_allocated[uid].get("gpu_specs", None), dict):
                            gpu_specs = self.stats_allocated[uid].get("gpu_specs", None)
                            score = self.stats_allocated[uid].get("score", 0)
                            self.stats[uid]["own_score"] = False
                        else:
                            gpu_specs = None
                            score = 0
                            self.stats[uid]["own_score"] = True
                    else:
                        score = 0
                        gpu_specs = None
                        self.stats[uid]["own_score"] = True

                if (
                    hotkey in penalized_hotkeys
                    or not isinstance(miner_details_all.get(hotkey), dict)
                    or not miner_details_all.get(hotkey)
                ):
                    score = 0

                self.stats[uid]["score"] = score*100
                self.stats[uid]["gpu_specs"] = gpu_specs

                # Keep or override reliability_score if you want
                if "reliability_score" not in self.stats[uid]:
                    self.stats[uid]["reliability_score"] = 0.0

            except KeyError as e:
                bt.logging.warning(f"KeyError occurred for UID {uid}: {str(e)}")
                score = 0
            except Exception as e:
                bt.logging.warning(f"An unexpected exception occurred for UID {uid}: {str(e)}")
                score = 0

            # Keep a simple reference of scores
            self.scores[uid] = score

        write_stats(self.db, self.stats)

        self.update_allocation_wandb()

        bt.logging.info("-" * 190)
        bt.logging.info("MINER STATS SUMMARY".center(190))
        bt.logging.info("-" * 190)

        for uid, data in self.stats.items():
            hotkey_str = str(data.get("hotkey", "unknown"))

            # Parse GPU specs into a human-readable format
            gpu_specs = data.get("gpu_specs")
            if isinstance(gpu_specs, dict):
                gpu_name = gpu_specs.get("gpu_name", "Unknown GPU")
                num_gpus = gpu_specs.get("num_gpus", 0)
                gpu_str = f"{num_gpus} x {gpu_name}" if num_gpus > 0 else "No GPUs"
            else:
                gpu_str = "N/A"  # Fallback if gpu_specs is not a dict

            # Format score as a float with 2 decimal digits
            raw_score = float(data.get("score", 0))
            score_str = f"{raw_score:.2f}"

            # Retrieve additional fields
            allocated = "yes" if data.get("allocated", False) else "no"
            reliability_score = data.get("reliability_score", 0)
            source = "Local" if data.get("own_score", False) else "External"

            # Format the log with fixed-width fields
            log_entry = (
                f"| UID: {uid:<4} | Hotkey: {hotkey_str:<45} | GPU: {gpu_str:<36} | "
                f"Score: {score_str:7} | Allocated: {allocated:<5} | "
                f"RelScore: {reliability_score:<5} | Source: {source:<9} |"
            )
            bt.logging.info(log_entry)

        # Add a closing dashed line
        bt.logging.info("-" * 190)

        bt.logging.info(f"🔢 Synced scores : {self.scores.tolist()}")

    def sync_local(self):
        """
        Resync our local state with the latest state from the blockchain.
        Sync scores with metagraph.
        Get the current uids of all miners in the network.
        """
        self.metagraph.sync(subtensor=self.subtensor)
        self.uids = self.metagraph.uids.tolist()

    def sync_status(self):
        # Check if the validator is still registered
        self.validator_subnet_uid = is_registered(
            wallet=self.wallet,
            metagraph=self.metagraph,
            subtensor=self.subtensor,
            entity="validator",
        )

        # Check for auto update
        if self.config.auto_update:
            try_update()

        # Check if the validator has the prometheus info updated
        subnet_prometheus_version = self.metagraph.neurons[self.validator_subnet_uid].prometheus_info.version
        current_version = __version_as_int__
        if subnet_prometheus_version != current_version:
            self.init_prometheus()

    def sync_miners_info(self, queryable_tuple_uids_axons: List[Tuple[int, bt.AxonInfo]]):
        if queryable_tuple_uids_axons:
            for uid, axon in queryable_tuple_uids_axons:
                if self.miners_items_to_set and (uid, axon.hotkey) not in self.miners_items_to_set:
                    try:
                        bt.logging.info(f"❌ Miner {uid}-{self.miners[uid]} has been deregistered. Clean up old entries.")
                        purge_miner_entries(self.db, uid, self.miners[uid])
                    except KeyError:
                        pass
                    bt.logging.info(f"✅ Setting up new miner {uid}-{axon.hotkey}.")
                    update_miners(self.db, [(uid, axon.hotkey)]),
                    self.miners[uid] = axon.hotkey
        else:
            bt.logging.warning(f"❌ No queryable miners.")

    @staticmethod
    def filter_axons(queryable_tuple_uids_axons: list[tuple[int, bt.AxonInfo]]) -> dict[int, bt.AxonInfo]:
        """Filter the axons with uids_list, remove those with the same IP address."""
        # FIXME(CSN-904): this does not work as intended, disabling till we know what to do
        bt.logging.debug("Axon filtering disabled")
        return dict(queryable_tuple_uids_axons)

        # Set to keep track of unique identifiers
        valid_ip_addresses = set()

        # List to store filtered axons
        dict_filtered_axons = {}
        for uid, axon in queryable_tuple_uids_axons:
            ip_address = axon.ip

            if ip_address not in valid_ip_addresses:
                valid_ip_addresses.add(ip_address)
                dict_filtered_axons[uid] = axon
            else:
                bt.logging.debug(f"Skipping duplicated IP UID: {uid}")

        return dict_filtered_axons

    def filter_axon_version(self, dict_filtered_axons: dict):
        # Get the minimal miner version
        latest_version = version2number(get_remote_version(pattern="__minimal_miner_version__"))
        if percent(len(dict_filtered_axons), self.total_current_miners) <= self.validator_whitelist_updated_threshold:
            bt.logging.info(f"Less than {self.validator_whitelist_updated_threshold}% miners are currently using the last version. Allowing all.")
            return dict_filtered_axons

        dict_filtered_axons_version = {}
        for uid, axon in dict_filtered_axons.items():
            if latest_version and latest_version <= axon.version:
                dict_filtered_axons_version[uid] = axon
            else:
                bt.logging.debug(f"Skipping outdated version UID: {uid}")
        return dict_filtered_axons_version

    def is_blacklisted(self, neuron: bt.NeuronInfoLite):
        coldkey = neuron.coldkey
        hotkey = neuron.hotkey

        # Blacklist coldkeys that are blacklisted by user
        if coldkey in self.blacklist_coldkeys:
            bt.logging.debug(f"Blacklisted recognized coldkey {coldkey} - with hotkey: {hotkey}")
            return True

        # Blacklist coldkeys that are blacklisted by user or by set of hotkeys
        if hotkey in self.blacklist_hotkeys:
            bt.logging.debug(f"Blacklisted recognized hotkey {hotkey}")
            # Add the coldkey attached to this hotkey in the blacklisted coldkeys
            self.blacklist_hotkeys.add(coldkey)
            return True

        # Blacklist coldkeys that are exploiters
        if coldkey in self.exploiters_coldkeys:
            bt.logging.debug(f"Blacklisted exploiter coldkey {coldkey} - with hotkey: {hotkey}")
            return True

        # Blacklist hotkeys that are exploiters
        if hotkey in self.exploiters_hotkeys:
            bt.logging.debug(f"Blacklisted exploiter hotkey {hotkey}")
            # Add the coldkey attached to this hotkey in the blacklisted coldkeys
            self.exploiters_hotkeys.add(coldkey)
            return True
        return False

    def get_valid_tensors(self, metagraph):
        tensors = []
        self.total_current_miners = 0
        for uid in metagraph.uids:
            neuron = metagraph.neurons[uid]

            if neuron.axon_info.ip != "0.0.0.0" and not self.is_blacklisted(neuron=neuron):
                self.total_current_miners += 1
                tensors.append(True)
            else:
                tensors.append(False)
        return tensors

    def get_valid_queryable(self):
        valid_queryable = []
        bt.logging.trace(f"All UIDs before filtering: {self.uids}")
        for uid in self.uids:
            neuron: bt.NeuronInfoLite = self.metagraph.neurons[uid]
            axon = self.metagraph.axons[uid]

            if neuron.axon_info.ip != "0.0.0.0" and not self.is_blacklisted(neuron=neuron):
                valid_queryable.append((uid, axon))
            elif self.is_blacklisted(neuron=neuron):
                bt.logging.trace(f"Skipping blacklisted UID: {uid}")
            else:
                bt.logging.trace(f"Skipping inactive UID: {uid}")

        bt.logging.trace(f"Valid UIDs after filtering: {[uid for uid, _ in valid_queryable]}")

        return valid_queryable

    def get_queryable(self):
        queryable = self.get_valid_queryable()

        # Execute a cleanup of the stats and miner information if the miner has been dereg
        self.sync_miners_info(queryable)

        dict_filtered_axons = self.filter_axons(queryable_tuple_uids_axons=queryable)
        dict_filtered_axons = self.filter_axon_version(dict_filtered_axons=dict_filtered_axons)
        return dict_filtered_axons

    def get_valid_validator_hotkeys(self):
        valid_uids = []
        uids = self.metagraph.uids.tolist()
        for index, uid in enumerate(uids):
            if self.metagraph.total_stake[index] > validator_permit_stake:
                valid_uids.append(uid)
        valid_hotkeys = []
        for uid in valid_uids:
            neuron = self.subtensor.neuron_for_uid(uid, self.config.netuid)
            hotkey = neuron.hotkey
            valid_hotkeys.append(hotkey)
        return valid_hotkeys

    async def get_specs_wandb(self):
        """
        Retrieves hardware specifications from Wandb, updates the miner_details table,
        and checks for differences in GPU specs, logging changes only for allocated hotkeys.
        Entries not present in Wandb will increment no_specs_count and be removed after 2 fails.
        """
        bt.logging.info(f"💻 Hardware list of uids queried (Wandb): {list(self._queryable_uids.keys())}")

        # Retrieve specs from Wandb
        specs_dict = self.wandb.get_miner_specs(self._queryable_uids)

        # Fetch current specs from miner_details using the existing function
        current_miner_details = get_miner_details(self.db)

        # Compare and detect GPU spec changes for allocated hotkeys
        for hotkey, new_specs in specs_dict.values():
            if hotkey in self.allocated_hotkeys:
                current_specs = current_miner_details.get(hotkey, {})
                current_gpu_specs = current_specs.get("gpu", {})
                new_gpu_specs = new_specs.get("gpu", {})

                # Extract the count values
                current_count = current_gpu_specs.get("count", 0)
                new_count = new_gpu_specs.get("count", 0)

                # Initialize names to None by default
                current_name = None
                new_name = None

                # Retrieve the current name if details are present and non-empty
                current_details = current_gpu_specs.get("details", [])
                if isinstance(current_details, list) and len(current_details) > 0:
                    current_name = current_details[0].get("name")

                # Retrieve the new name if details are present and non-empty
                new_details = new_gpu_specs.get("details", [])
                if isinstance(new_details, list) and len(new_details) > 0:
                    new_name = new_details[0].get("name")

                # Compare only count and name
                if current_count != new_count or current_name != new_name:
                    axon = None
                    for uid, axon_info in self._queryable_uids.items():
                        if axon_info.hotkey == hotkey:
                            axon = axon_info
                            break

                    if axon:
                        bt.logging.info(f"GPU specs changed for allocated hotkey {hotkey}:")
                        bt.logging.info(f"Old count: {current_count}, Old name: {current_name}")
                        bt.logging.info(f"New count: {new_count}, New name: {new_name}")
                        await self.deallocate_miner(axon, None)

        # Collect the hotkeys present in Wandb this pass
        present_hotkeys = {hk for (hk, _specs) in specs_dict.values()}

        # Update the local db with the new data from Wandb
        update_miner_details(self.db, present_hotkeys, list(specs_dict.values()))

        self.finalized_specs_once = True

    async def proof_of_gpu(self):
        """
        Perform Proof-of-GPU benchmarking on allocated miners without overlapping tests.
        Uses asyncio with ThreadPoolExecutor to test miners in parallel.
        """
        try:
            # Init miners to be tested
            self._queryable_uids = self.get_queryable()
            valid_validator_hotkeys = self.get_valid_validator_hotkeys()
            self.allocated_hotkeys = self.wandb.get_allocated_hotkeys(valid_validator_hotkeys, True)

            # Settings
            merkle_proof = self.config_data["merkle_proof"]
            retry_limit = merkle_proof.get("pog_retry_limit",30)
            retry_interval = merkle_proof.get("pog_retry_interval",75)
            num_workers = merkle_proof.get("max_workers",32)
            max_delay = merkle_proof.get("max_random_delay",1200)

            # Random delay for PoG
            delay = random.uniform(0, max_delay)  # Random delay
            bt.logging.info(f"💻⏳ Scheduled Proof-of-GPU task to start in {delay:.2f} seconds.")
            await asyncio.sleep(delay)

            bt.logging.info(f"💻 Starting Proof-of-GPU benchmarking for uids: {list(self._queryable_uids.keys())}")
            # Shared dictionary to store results
            self.results = {}
            # Dictionary to track retry counts
            retry_counts = defaultdict(int)
            # Queue of miners to process
            queue = asyncio.Queue()

            # Initialize the queue with initial miners
            for i in range(0, len(self.uids), self.validator_challenge_batch_size):
                for _uid in self.uids[i : i + self.validator_challenge_batch_size]:
                    try:
                        axon = self._queryable_uids[_uid]
                        if axon.hotkey in self.allocated_hotkeys:
                            bt.logging.info(f"Skipping allocated miner: {axon.hotkey}")
                            continue  # skip this miner since it's allocated
                        await queue.put(axon)
                    except KeyError:
                        continue

            # Initialize a single Lock for thread-safe updates to results
            results_lock = asyncio.Lock()

            async def worker():
                while True:
                    try:
                        axon = await queue.get()
                    except asyncio.CancelledError:
                        break
                    hotkey = axon.hotkey
                    try:
                        # Set a timeout for the GPU test
                        timeout = 300  # e.g., 5 minutes
                        # Define a synchronous helper function to run the asynchronous test_miner_gpu
                        # This is required because run_in_executor expects a synchronous callable.
                        def run_test_miner_gpu():
                            # Run the async test_miner_gpu function and wait for its result.
                            future = asyncio.run_coroutine_threadsafe(self.test_miner_gpu(axon, self.config_data), self.loop)
                            return future.result()

                        # Submit the run_test_miner_gpu function to a thread pool executor.
                        # The asyncio.wait_for is used to enforce a timeout for the overall operation.
                        result = await asyncio.wait_for(
                            asyncio.get_running_loop().run_in_executor(
                                self.executor, run_test_miner_gpu
                            ),
                            timeout=timeout
                        )
                        if result[1] is not None and result[2] > 0:
                            async with results_lock:
                                self.results[hotkey] = {
                                    "gpu_name": result[1],
                                    "num_gpus": result[2]
                                }
                            update_pog_stats(self.db, hotkey, result[1], result[2])
                        elif result[1] is None and result[2] == -1:
                            # Health check failed - don't retry
                            bt.logging.info(f"❌ {hotkey}: Health check failed, skipping retry")
                            update_pog_stats(self.db, hotkey, None, None)
                        else:
                            raise RuntimeError("GPU test failed")
                    except asyncio.TimeoutError:
                        bt.logging.warning(f"⏳ Timeout while testing {hotkey}. Retrying...")
                        retry_counts[hotkey] += 1
                        if retry_counts[hotkey] < retry_limit:
                            bt.logging.info(f"🔄 {hotkey}: Retrying miner -> (Attempt {retry_counts[hotkey]})")
                            await asyncio.sleep(retry_interval)
                            await queue.put(axon)
                        else:
                            bt.logging.info(f"❌ {hotkey}: Miner failed after {retry_limit} attempts (Timeout).")
                            update_pog_stats(self.db, hotkey, None, None)
                    except Exception as e:
                        bt.logging.debug(f"Exception in worker for {hotkey}: {e}")
                        retry_counts[hotkey] += 1
                        if retry_counts[hotkey] < retry_limit:
                            bt.logging.info(f"🔄 {hotkey}: Retrying miner -> (Attempt {retry_counts[hotkey]})")
                            await asyncio.sleep(retry_interval)
                            await queue.put(axon)
                        else:
                            bt.logging.info(f"❌ {hotkey}: Miner failed after {retry_limit} attempts.")
                            update_pog_stats(self.db, hotkey, None, None)
                    finally:
                        queue.task_done()

            # Number of concurrent workers
            # Determine a safe default number of workers
            cpu_cores = os.cpu_count() or 1
            safe_max_workers = min((cpu_cores + 4)*4, num_workers)

            workers = [asyncio.create_task(worker()) for _ in range(safe_max_workers)]
            bt.logging.debug(f"Started {safe_max_workers} worker tasks for Proof-of-GPU benchmarking.")

            # Wait until the queue is fully processed
            await queue.join()

            # Cancel worker tasks
            for w in workers:
                w.cancel()
            # Wait until all worker tasks are cancelled
            await asyncio.gather(*workers, return_exceptions=True)

            bt.logging.success(f"✅ Proof-of-GPU benchmarking completed.")
            return self.results
        except Exception as e:
            bt.logging.info(f"❌ Exception in proof_of_gpu: {e}\n{traceback.format_exc()}")

    def on_gpu_task_done(self, task):
        try:
            results = task.result()
            bt.logging.debug(f"Proof-of-GPU Results: {results}")
            self.gpu_task = None  # Reset the task reference
            self.sync_scores()

        except Exception as e:
            bt.logging.error(f"Proof-of-GPU task failed: {e}")
            self.gpu_task = None

    async def _publish_pog_result_event(
        self, hotkey, request_id, start_time, result,
        benchmark_data: dict | None = None,
        health_check_result: bool | None = None,
        error_details: str | None = None,
    ):
        # Publish successful POG result
        validation_duration = time.time() - start_time
        await self.pubsub_client.publish_pog_result_event(
            miner_hotkey=hotkey,
            request_id=request_id,
            result=result,
            validation_duration=validation_duration,
            benchmark_data=benchmark_data,
            health_check_result=health_check_result,
            error_details=error_details
        )

    async def test_miner_gpu(self, axon, config_data):
        """
        Allocate, test, and deallocate a single miner (Sybil-compatible).
        :return: Tuple of (miner_hotkey, gpu_name, num_gpus)
        """
        allocation_status = False
        miner_info = None
        host = None
        hotkey = axon.hotkey
        request_id = str(uuid.uuid4())
        start_time = time.time()
        public_key = None
        bt.logging.info(f"{hotkey}: Starting miner test.")

        try:
            gpu_data = config_data["gpu_performance"]
            gpu_tolerance_pairs = gpu_data.get("gpu_tolerance_pairs", {})
            merkle_proof = config_data["merkle_proof"]
            time_tol = merkle_proof.get("time_tolerance", 5)
            # Extract miner_script path
            miner_script_path = merkle_proof["miner_script_path"]

            # Step 1: Allocate Miner
            private_key, public_key = rsa.generate_key_pair()
            allocation_response = await self.allocate_miner(axon, private_key, public_key)
            if not allocation_response:
                bt.logging.trace(f"🌀 {hotkey}: Busy or not allocatable.")
                return (hotkey, None, 0)
            allocation_status = True
            miner_info = allocation_response
            host = miner_info['host']
            bt.logging.trace(f"{hotkey}: Allocated Miner for testing.")

            # Step 2: Connect via SSH
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            bt.logging.trace(f"{hotkey}: Connect to Miner via SSH.")
            ssh_client.connect(
                host,
                port=miner_info.get('port', 22),
                username=miner_info['username'],
                password=miner_info['password'],
                timeout=10,
            )
            if not (ssh_client):
                ssh_client.close()
                bt.logging.trace(f"{hotkey}: SSH connection failed.")
                return (hotkey, None, -1)
            bt.logging.trace(f"{hotkey}: Connected to Miner via SSH.")

            # Step 3: Hash Check
            local_hash = compute_script_hash(miner_script_path)
            bt.logging.trace(f"{hotkey}: [Step 1] Local script hash computed successfully.")
            bt.logging.trace(f"{hotkey}: Local Hash: {local_hash}")
            remote_hash = send_script_and_request_hash(ssh_client, miner_script_path)
            bt.logging.trace(f"{hotkey}: [Step 1] Remote script hash received.")
            bt.logging.trace(f"{hotkey}: Remote Hash: {remote_hash}")
            if local_hash != remote_hash:
                bt.logging.trace(f"{hotkey}: [Integrity Check] FAILURE: Hash mismatch detected.")
                raise ValueError(f"{hotkey}: Script integrity verification failed.")

            # Step 4: Get GPU info NVIDIA from the remote miner
            bt.logging.trace(f"{hotkey}: [Step 4] Retrieving GPU information (NVIDIA driver) from miner...")
            gpu_info = get_remote_gpu_info(ssh_client)
            num_gpus_reported = gpu_info["num_gpus"]
            gpu_name_reported = gpu_info["gpu_names"][0] if num_gpus_reported > 0 else None
            bt.logging.debug(f"{hotkey}: [Step 4] Reported GPU Information:")
            if num_gpus_reported > 0:
                bt.logging.debug(f"{hotkey}: Number of GPUs: {num_gpus_reported}")
                bt.logging.debug(f"{hotkey}: GPU Type: {gpu_name_reported}")
            if num_gpus_reported <= 0:
                bt.logging.debug(f"{hotkey}: No GPUs detected.")
                raise ValueError("No GPUs detected.")

            # Step 5: Run the benchmarking mode
            bt.logging.debug(f"💻 {hotkey}: Executing benchmarking mode.")
            bt.logging.trace(f"{hotkey}: [Step 5] Executing benchmarking mode on the miner...")
            execution_output = execute_script_on_miner(ssh_client, mode='benchmark')
            bt.logging.trace(f"{hotkey}: [Step 5] Benchmarking completed.")
            # Parse the execution output (Sybil compatible)
            num_gpus, vram, size_fp16, time_fp16, size_fp32, time_fp32 = parse_benchmark_output(execution_output)
            bt.logging.trace(f"{hotkey}: [Benchmark Results] Detected {num_gpus} GPU(s) with {vram} GB unfractured VRAM.")
            bt.logging.trace(f"{hotkey}: FP16 - Matrix Size: {size_fp16}, Execution Time: {time_fp16} s")
            bt.logging.trace(f"{hotkey}: FP32 - Matrix Size: {size_fp32}, Execution Time: {time_fp32} s")
            # Calculate performance metrics
            fp16_tflops = (2 * size_fp16 ** 3) / time_fp16 / 1e12
            fp32_tflops = (2 * size_fp32 ** 3) / time_fp32 / 1e12
            bt.logging.trace(f"{hotkey}: [Performance Metrics] Calculated TFLOPS:")
            bt.logging.trace(f"{hotkey}: FP16: {fp16_tflops:.2f} TFLOPS")
            bt.logging.trace(f"{hotkey}: FP32: {fp32_tflops:.2f} TFLOPS")
            gpu_name = identify_gpu(fp16_tflops, fp32_tflops, vram, gpu_data, gpu_name_reported, gpu_tolerance_pairs)
            bt.logging.trace(f"{hotkey}: [GPU Identification] Based on performance: {gpu_name}")
            # Step 5.1: Ensure reported GPU matches benchmark-identified GPU
            if gpu_name != gpu_name_reported:
                bt.logging.debug(f"{hotkey}: GPU mismatch! Reported: {gpu_name_reported}, Benchmarked: {gpu_name}")
                raise ValueError("Reported GPU and benchmark GPU do not match.")

            # Step 6: Run the Merkle proof mode
            bt.logging.debug(f"{hotkey}: [Step 6] Initiating Merkle Proof Mode.")
            # Step 1: Send seeds and execute compute mode
            n = adjust_matrix_size(vram, element_size=4, buffer_factor=0.05)
            seeds = get_random_seeds(num_gpus)
            send_seeds(ssh_client, seeds, n)
            bt.logging.trace(f"{hotkey}: [Step 6] Compute mode executed on miner - Matrix Size: {n}")
            start_time = time.time()
            execution_output = execute_script_on_miner(ssh_client, mode='compute')
            end_time = time.time()
            elapsed_time = end_time - start_time
            bt.logging.trace(f"{hotkey}: Compute mode execution time: {elapsed_time:.2f} seconds.")
            # Parse the execution output (Sybil compatible)
            root_hashes_list, gpu_timings_list = parse_merkle_output(execution_output)
            bt.logging.trace(f"{hotkey}: [Merkle Proof] Root hashes received from GPUs:")
            for gpu_id, root_hash in root_hashes_list:
                bt.logging.trace(f"{hotkey}: GPU {gpu_id}: {root_hash}")

            # Calculate total times
            total_multiplication_time = 0.0
            total_merkle_tree_time = 0.0
            num_gpus = len(gpu_timings_list)
            for _, timing in gpu_timings_list:
                total_multiplication_time += timing.get('gemm', 0.0)
                total_merkle_tree_time += timing.get('merkle', 0.0)
            average_multiplication_time = total_multiplication_time / num_gpus if num_gpus > 0 else 0.0
            average_merkle_tree_time = total_merkle_tree_time / num_gpus if num_gpus > 0 else 0.0
            bt.logging.trace(f"{hotkey}: Average Matrix Multiplication Time: {average_multiplication_time:.4f} seconds")
            bt.logging.trace(f"{hotkey}: Average Merkle Tree Time: {average_merkle_tree_time:.4f} seconds")

            timing_passed = False
            if elapsed_time < time_tol + num_gpus * time_fp32 and average_multiplication_time < time_fp32:
                timing_passed = True

            # Step 7: Verify merkle proof
            root_hashes = {gpu_id: root_hash for gpu_id, root_hash in root_hashes_list}
            gpu_timings = {gpu_id: timing for gpu_id, timing in gpu_timings_list}
            n = gpu_timings[0]['n'] if 0 in gpu_timings else n
            indices = {}
            num_indices = 1
            for gpu_id in range(num_gpus):
                indices[gpu_id] = [(np.random.randint(0, 2 * n), np.random.randint(0, n)) for _ in range(num_indices)]
            send_challenge_indices(ssh_client, indices)
            execution_output = execute_script_on_miner(ssh_client, mode='proof')
            bt.logging.debug(f"{hotkey}: [Merkle Proof] Proof mode executed on miner.")
            responses = receive_responses(ssh_client, num_gpus)
            bt.logging.trace(f"{hotkey}: [Merkle Proof] Responses received from miner.")

            verification_passed = verify_responses(seeds, root_hashes, responses, indices, n)
            if verification_passed and timing_passed:
                bt.logging.success(f"✅ {hotkey}: GPU Identification: Detected {num_gpus} x {gpu_name} GPU(s)")

                # Step 8: Perform health check on the same miner after POG is successful
                bt.logging.debug(f"🏥 {hotkey}: POG completed successfully, starting health check...")
                bt.logging.trace(f"{hotkey}: [Step 8] Initiating health check...")
                try:
                    health_check_result = perform_health_check(axon, miner_info)
                    if health_check_result:
                        bt.logging.success(f"✅ {hotkey}: Health check passed")
                        bt.logging.trace(f"{hotkey}: [Step 8] Health check completed successfully - miner is accessible")
                        await self._publish_pog_result_event(
                            hotkey=hotkey,
                            request_id=request_id,
                            start_time=start_time,
                            result="success",
                            benchmark_data={
                                "reported_gpu_number": num_gpus_reported,
                                "reported_gpu_name": gpu_name_reported,
                                "vram": vram,
                                "size_fp16": size_fp16,
                                "time_fp16": time_fp16,
                                "size_fp32": size_fp32,
                                "time_fp32": time_fp32,
                                "fp16_tflops": fp16_tflops,
                                "fp32_tflops": fp32_tflops,
                                "identified_gpu_number": num_gpus,
                                "identified_gpu_name": gpu_name,
                                "average_multiplication_time": average_multiplication_time,
                                "average_merkle_tree_time": average_merkle_tree_time,
                                "verification_passed": verification_passed,
                                "timing_passed": timing_passed,
                            },
                            health_check_result=health_check_result
                        )
                        return (hotkey, gpu_name, num_gpus)
                    else:
                        bt.logging.debug(f"⚠️ {hotkey}: Health check failed")
                        bt.logging.trace(f"{hotkey}: [Step 8] Health check failed - miner is not accessible")
                        bt.logging.info(f"⚠️ {hotkey}: GPU Identification: Aborted due to health check failure")
                        await self._publish_pog_result_event(
                            hotkey=hotkey,
                            request_id=request_id,
                            start_time=start_time,
                            result='failure',
                            error_details='Health check failed',
                            health_check_result=False,
                            benchmark_data={
                                "reported_gpu_number": num_gpus_reported,
                                "reported_gpu_name": gpu_name_reported,
                                "vram": vram,
                                "size_fp16": size_fp16,
                                "time_fp16": time_fp16,
                                "size_fp32": size_fp32,
                                "time_fp32": time_fp32,
                                "fp16_tflops": fp16_tflops,
                                "fp32_tflops": fp32_tflops,
                                "identified_gpu_number": num_gpus,
                                "identified_gpu_name": gpu_name,
                                "average_multiplication_time": average_multiplication_time,
                                "average_merkle_tree_time": average_merkle_tree_time,
                                "verification_passed": verification_passed,
                                "timing_passed": timing_passed,
                            },
                        )
                        return (hotkey, None, -1)  # Use -1 to indicate health check failure
                except Exception as e:
                    bt.logging.debug(f"❌ {hotkey}: Error during health check: {e}")
                    bt.logging.trace(f"{hotkey}: [Step 8] Health check error: {e}")
                    bt.logging.info(f"⚠️ {hotkey}: GPU Identification: Aborted due to health check error")
                    await self._publish_pog_result_event(
                        hotkey=hotkey,
                        request_id=request_id,
                        start_time=start_time,
                        result='error',
                        error_details=f'Health check failed: {str(e)}',
                        benchmark_data={
                            "reported_gpu_number": num_gpus_reported,
                            "reported_gpu_name": gpu_name_reported,
                            "vram": vram,
                            "size_fp16": size_fp16,
                            "time_fp16": time_fp16,
                            "size_fp32": size_fp32,
                            "time_fp32": time_fp32,
                            "fp16_tflops": fp16_tflops,
                            "fp32_tflops": fp32_tflops,
                            "identified_gpu_number": num_gpus,
                            "identified_gpu_name": gpu_name,
                            "average_multiplication_time": average_multiplication_time,
                            "average_merkle_tree_time": average_merkle_tree_time,
                            "verification_passed": verification_passed,
                            "timing_passed": timing_passed,
                        },
                        health_check_result=False,
                    )
                    return (hotkey, None, -1)  # Use -1 to indicate health check failure
            else:
                bt.logging.info(f"⚠️  {hotkey}: GPU Identification: Aborted due to verification failure (verification={verification_passed}, timing={timing_passed})")
                await self._publish_pog_result_event(
                    hotkey=hotkey,
                    request_id=request_id,
                    start_time=start_time,
                    result='failure',
                    error_details='Verification or timing failed',
                    benchmark_data={
                        "reported_gpu_number": num_gpus_reported,
                        "reported_gpu_name": gpu_name_reported,
                        "vram": vram,
                        "size_fp16": size_fp16,
                        "time_fp16": time_fp16,
                        "size_fp32": size_fp32,
                        "time_fp32": time_fp32,
                        "fp16_tflops": fp16_tflops,
                        "fp32_tflops": fp32_tflops,
                        "identified_gpu_number": num_gpus,
                        "identified_gpu_name": gpu_name,
                        "average_multiplication_time": average_multiplication_time,
                        "average_merkle_tree_time": average_merkle_tree_time,
                        "verification_passed": verification_passed,
                        "timing_passed": timing_passed,
                    },
                    health_check_result=False,
                )
                return (hotkey, None, 0)

        except Exception as e:
            bt.logging.debug(f"❌ {hotkey}: Error testing Miner: {e}", exc_info=True)
            await self._publish_pog_result_event(
                hotkey=hotkey,
                request_id=request_id,
                start_time=start_time,
                result='error',
                error_details=f'Testing miner failed: {str(e)}',
            )
            return (hotkey, None, 0)

        finally:
            try:
                if ssh_client:
                    ssh_client.close()
            except Exception:
                pass
            try:
                if allocation_status and miner_info:
                    await self.deallocate_miner(axon, public_key)
                    bt.logging.trace(f"{hotkey}: Miner de-allocated.")
            except Exception as e:
                bt.logging.debug(f"{hotkey}: Miner de-allocation failed: {e}")

    async def allocate_miner(
        self,
        axon: bt.AxonInfo,
        private_key: str,
        public_key: str,
    ) -> dict | None:
        """
        Ask the allocator on ``axon`` for one container and return SSH creds.

        • No preliminary “checking=True” probe – we directly request the slot.
        • Retries up to 5× on transient disconnects with linear back-off (1 s, 2 s, 3 s, 4 s).
        • Returns *None* if the miner is busy/declined or all retries fail.
        """
        device_requirement = {
            "cpu":       {"count": 1},
            "gpu":       {"count": 1, "capacity": 0, "type": ""},
            "hard_disk": {"capacity": 1_073_741_824},   # 1 GiB
            "ram":       {"capacity": 1_073_741_824},   # 1 GiB
            "testing":   True,
        }
        docker_requirement = {
            "base_image": "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime",
        }

        MAX_TRIES      = 5
        BASE_BACKOFF_S = 1  # 1 s, 2 s, 3 s, 4 s

        for attempt in range(1, MAX_TRIES + 1):
            try:
                async with bt.dendrite(wallet=self.wallet) as dendrite:
                    rsp = await dendrite(
                        axon,
                        Allocate(
                            timeline=1,                    # one-shot job
                            device_requirement=device_requirement,
                            checking=False,               # real allocation
                            public_key=public_key,
                            docker_requirement=docker_requirement,
                        ),
                        timeout=60,
                    )

                    if rsp and rsp.get("status", False):
                        # ---- decrypt allocator’s reply -----------------------
                        dec  = rsa.decrypt_data(
                            private_key.encode(),
                            base64.b64decode(rsp["info"]),
                        )
                        info = json.loads(dec)

                        miner_info = {
                            'host': axon.ip,
                            'port': info['port'],
                            'username': info['username'],
                            'password': info['password'],
                            'external_user_ports': info.get('external_user_ports', {}),
                        }
                        await self.pubsub_client.publish_miner_allocation(
                            miner_hotkey=axon.hotkey,
                            allocation_result=True,
                        )
                        bt.logging.trace(f"Successfully allocated miner {axon.hotkey}")
                        return miner_info

                    # allocator politely said “busy” or returned invalid status
                    else:
                        if not rsp:
                            bt.logging.trace(f"{axon.hotkey}: No response received for miner allocation.")
                        else:
                            bt.logging.trace(f"{axon.hotkey}: Miner allocation request failed.")
                            bt.logging.trace(f"{axon.hotkey}: Miner allocation response: {rsp}")

                        await self.pubsub_client.publish_miner_allocation(
                            miner_hotkey=axon.hotkey,
                            allocation_result=False,
                            allocation_error=(
                                'No response received'
                                if not rsp
                                else 'Miner allocation request failed'
                            ),
                        )
                        return None

            # -------- transient disconnects / 503 ------------------------------
            except bt.dendrite.exceptions.ServerDisconnectedError as e:
                bt.logging.warning(
                    f"{axon.hotkey}: allocator disconnected "
                    f"(attempt {attempt}/{MAX_TRIES}) – {e}"
                )
                await self.pubsub_client.publish_miner_allocation(
                    miner_hotkey=axon.hotkey,
                    allocation_result=False,
                    allocation_error="Allocator disconnected",
                )
            except ConnectionRefusedError as e:
                bt.logging.warning(
                    f"{axon.hotkey}: connection refused "
                    f"(attempt {attempt}/{MAX_TRIES}) – {e}"
                )
                await self.pubsub_client.publish_miner_allocation(
                    miner_hotkey=axon.hotkey,
                    allocation_result=False,
                    allocation_error="Connection refused during miner allocation",
                )
            # -------- any other error → give up immediately --------------------
            except Exception as e:
                bt.logging.trace(f"{axon.hotkey}: allocation exception – {e}")
                await self.pubsub_client.publish_miner_allocation(
                    miner_hotkey=axon.hotkey,
                    allocation_result=False,
                    allocation_error=f'Miner allocation failed: {str(e)}'
                )
                return None

            # back-off before next retry for transient errors
            if attempt < MAX_TRIES:
                await asyncio.sleep(BASE_BACKOFF_S * attempt)

        # all retries exhausted
        await self.pubsub_client.publish_miner_allocation(
            miner_hotkey=axon.hotkey,
            allocation_result=False,
            allocation_error="All retries exhausted",
        )
        return None

    async def deallocate_miner(self, axon, public_key):
        """
        Deallocate a miner by sending a deregistration query.

        :param axon: Axon object containing miner details.
        :param public_key: Public key of the miner; if None, it will be retrieved from the database.
        """
        deallocation_error = None

        if not public_key:
            try:
                # Instantiate the connection to the database and retrieve miner details
                db = ComputeDb()
                cursor = db.get_cursor()

                cursor.execute(
                    "SELECT details, hotkey FROM allocation WHERE hotkey = ?",
                    (axon.hotkey,)
                )
                row = cursor.fetchone()

                if row:
                    info = json.loads(row[0])  # Parse JSON string from the 'details' column
                    public_key = info.get("regkey")
            except Exception as e:
                deallocation_error = str(e)
                bt.logging.trace(f"{axon.hotkey}: Missing public key: {e}")

        try:
            retry_count = 0
            max_retries = 3
            allocation_status = True

            while allocation_status and retry_count < max_retries:
                try:
                    async with bt.dendrite(wallet=self.wallet) as dendrite:
                        # Send deallocation query
                        deregister_response = await dendrite(
                            axon,
                            Allocate(
                                timeline=0,
                                checking=False,
                                public_key=public_key,
                            ),
                            timeout=15,
                        )

                        if deregister_response and deregister_response.get("status") is True:
                            allocation_status = False
                            bt.logging.trace(f"Deallocated miner {axon.hotkey}")
                        else:
                            retry_count += 1
                            bt.logging.trace(
                                f"{axon.hotkey}: Failed to deallocate miner. "
                                f"(attempt {retry_count}/{max_retries})"
                            )
                            if retry_count >= max_retries:
                                bt.logging.trace(f"{axon.hotkey}: Max retries reached for deallocating miner.")
                            await asyncio.sleep(5)
                except Exception as e:
                    retry_count += 1
                    deallocation_error = str(e)
                    bt.logging.trace(
                        f"{axon.hotkey}: Error while trying to deallocate miner. "
                        f"(attempt {retry_count}/{max_retries}): {e}"
                    )
                    if retry_count >= max_retries:
                        bt.logging.trace(f"{axon.hotkey}: Max retries reached for deallocating miner.")
                    await asyncio.sleep(5)
        except Exception as e:
            deallocation_error = str(e)
            bt.logging.trace(f"{axon.hotkey}: Unexpected error during deallocation: {e}")

        await self.pubsub_client.publish_miner_deallocation(
            miner_hotkey=axon.hotkey,
            retry_count=retry_count,
            deallocation_result=allocation_status is False,
            deallocation_error=deallocation_error
        )

    def get_burn_uid(self) -> int:
        """
        Returns the UID of the subnet owner (the burn account) for this subnet.
        """
        # 1) Query the on-chain SubnetOwner hotkey
        sn_owner_hotkey = self.subtensor.query_subtensor(
            "SubnetOwnerHotkey",
            params=[self.config.netuid],
        )
        bt.logging.info(f"Subnet Owner Hotkey: {sn_owner_hotkey}")

        # 2) Convert that hotkey to its UID on this subnet
        burn_uid = self.subtensor.get_uid_for_hotkey_on_subnet(
            hotkey_ss58=sn_owner_hotkey,
            netuid=self.config.netuid,
        )
        bt.logging.info(f"Subnet Owner UID (burn): {burn_uid}")
        return burn_uid

    def set_burn_weights(self):
        """
        Assigns 100% of the weight to the burn UID by clamping negatives → 0,
        L1-normalizing [1.0] into a weight, and pushing on-chain.
        """
        # 1) fetch burn UID
        burn_uid = self.get_burn_uid()

        # 2) prepare a single-element score tensor
        scores = torch.tensor([1.0], dtype=torch.float32)
        scores[scores < 0] = 0

        # 3) normalize into a weight vector that sums to 1
        weights: torch.FloatTensor = torch.nn.functional.normalize(scores, p=1.0, dim=0).float()
        bt.logging.info(f"🔥 Burn-only weight: {weights.tolist()}")

        # 4) send to chain
        result = self.subtensor.set_weights(
            netuid=self.config.netuid,
            wallet=self.wallet,
            uids=[burn_uid],
            weights=weights,
            version_key=__version_as_int__,
            wait_for_inclusion=False,
        )

        if isinstance(result, tuple) and result[0]:
            bt.logging.success("✅ Successfully set burn weights.")
        else:
            bt.logging.error(f"❌ Failed to set burn weights: {result}")

    def set_weight_capped_by_gpu(self):
        """
        Distribute emission weights to miners based on GPU type priorities (normalized),
        capped by total_miner_emission. Remaining weight is burned.
        """
        try:
            # Load config
            subnet_config = self.config_data.get("subnet_config", {})
            total_miner_emission = float(subnet_config.get("total_miner_emission", 0.0))
            gpu_priorities = subnet_config.get("gpu_weights", {})

            # Clamp emission
            total_miner_emission = min(max(total_miner_emission, 0.0), 1.0)

            # Prepare miner data
            uid_to_gpu = {}
            uid_to_score = {}
            gpu_groups = {}

            for uid in self.uids:
                if uid not in self.stats:
                    continue

                stats_entry = self.stats[uid]
                score = max(0.0, float(stats_entry.get("score", 0.0))) / 100.0

                gpu_specs = stats_entry.get("gpu_specs", {})
                if not isinstance(gpu_specs, dict):
                    continue

                gpu_name = gpu_specs.get("gpu_name", None)
                if gpu_name is None or gpu_name not in gpu_priorities:
                    continue

                priority = gpu_priorities[gpu_name]
                if priority <= 0:
                    continue

                uid_to_gpu[uid] = gpu_name
                uid_to_score[uid] = score

                if gpu_name not in gpu_groups:
                    gpu_groups[gpu_name] = []
                gpu_groups[gpu_name].append(uid)

            # Normalize GPU priorities
            total_priority = sum(v for v in gpu_priorities.values() if v > 0)
            if total_priority == 0:
                bt.logging.warning("⚠️ All GPU priorities are 0. Entire emission will be burned.")
                total_assigned_weight = 0.0
                uid_weights = torch.zeros(len(self.uids), dtype=torch.float32)
            else:
                uid_weights = torch.zeros(len(self.uids), dtype=torch.float32)
                total_assigned_weight = 0.0
                gpu_actual_emission = {}

                for gpu_name, uids in gpu_groups.items():
                    priority = gpu_priorities[gpu_name]
                    group_cap = (priority / total_priority) * total_miner_emission

                    scores = torch.tensor([uid_to_score[uid] for uid in uids], dtype=torch.float32)
                    if scores.sum() == 0:
                        continue  # no emission for this group

                    normalized = scores / scores.sum()
                    capped = normalized * group_cap

                    for i, uid in enumerate(uids):
                        idx = self.uids.index(uid)
                        uid_weights[idx] = capped[i]

                    gpu_actual_emission[gpu_name] = group_cap
                    total_assigned_weight += group_cap

            # Burn the rest
            burn_uid    = self.get_burn_uid()
            burn_weight = max(0.0, 1.0 - total_assigned_weight)

            if burn_uid in self.uids:
                # burn UID already among miners → overwrite its slot
                uids    = self.uids                      # keep original order
                weights = uid_weights.clone()            # same length
                idx     = self.uids.index(burn_uid)
                weights[idx] = burn_weight
                bt.logging.debug("[Weights] burn_uid overwritten in-place")
            else:
                # burn UID not present → append it
                uids    = self.uids + [burn_uid]
                weights = torch.cat(
                    [uid_weights, torch.tensor([burn_weight], dtype=torch.float32)]
                )
                bt.logging.debug("[Weights] burn_uid appended")

            # final normalisation guard
            weights = weights / weights.sum()

            # Logging
            # Debug breakdown per GPU
            bt.logging.debug("📊 Emission breakdown per GPU group:")

            # GPU group emissions
            for gpu_name, cap in gpu_actual_emission.items():
                percent = cap * 100.0
                bt.logging.debug(f"   • {gpu_name:<25} {percent:6.2f}%")

            # Burned portion
            burn_percent = burn_weight * 100.0

            # Totals
            bt.logging.info(f"📈 Total miner emission:       {(total_assigned_weight * 100):6.2f}%")
            bt.logging.info(f"🔥 Burned emission:            {burn_percent:6.2f}%")
            bt.logging.info(f"⚙️ Final weights: {weights.tolist()}")

            result = self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=uids,
                weights=weights,
                version_key=__version_as_int__,
                wait_for_inclusion=False,
            )

            if isinstance(result, tuple) and result[0]:
                bt.logging.success("✅ Successfully set capped GPU-based weights.")
            else:
                bt.logging.error(f"❌ Failed to set GPU-capped weights: {result}")

        except Exception as e:
            bt.logging.error(f"❌ Exception in set_weight_capped_by_gpu: {e}")

    def set_weights(self):
        # Remove all negative scores and attribute them 0.
        self.scores[self.scores < 0] = 0
        # Normalize the scores into weights
        weights: torch.FloatTensor = torch.nn.functional.normalize(self.scores, p=1.0, dim=0).float()
        bt.logging.info(f"🏋️ Weight of miners : {weights.tolist()}")
        # This is a crucial step that updates the incentive mechanism on the Bittensor blockchain.
        # Miners with higher scores (or weights) receive a larger share of TAO rewards on this subnet.
        result = self.subtensor.set_weights(
            netuid=self.config.netuid,  # Subnet to set weights on.
            wallet=self.wallet,  # Wallet to sign set weights using hotkey.
            uids=self.uids,  # Uids of the miners to set weights for.
            weights=weights,  # Weights to set for the miners.
            version_key=__version_as_int__,
            wait_for_inclusion=False,
        ) # return type: tuple[bool, str]
        if isinstance(result, tuple) and result and isinstance(result[0], bool) and result[0]:
            bt.logging.info(result)
            bt.logging.success("✅ Successfully set weights.")
        else:
            bt.logging.error(result)
            bt.logging.error("❌ Failed to set weights.")

    def next_info(self, cond, next_block):
        if cond:
            return calculate_next_block_time(self.current_block, next_block)
        else:
            return None

    def current_epoch(self, blk: int | None = None) -> int:
        return (blk or self.current_block) // self.blocks_per_epoch

    def epoch_is_pog(self, ep: int | None = None) -> bool:
        return (ep if ep is not None else self.current_epoch()) % 2 == 0

    def epoch_start_block(self, ep: int | None = None) -> int:
        e = ep if ep is not None else self.current_epoch()
        return e * self.blocks_per_epoch

    def my_sybil_slot(self, ep: int | None = None) -> tuple[int, int]:
        """
        Return (slot_start, slot_end) for the Sybil-PoG epoch.
        If our hotkey isn’t in the on-chain validator set and allow_fake_sybil_slot=False,
        Returns slot 0 otherwise.
        """
        ep    = ep if ep is not None else self.current_epoch()
        start = self.epoch_start_block(ep)

        # ─── grab validator set and our own hotkey ────────────────────────
        vals = sorted(self.get_valid_validator_hotkeys())

        hotkey_obj = self.wallet.hotkey
        # Keypair itself vs actual ss58 string:
        my_hk_obj = hotkey_obj
        my_hk     = getattr(hotkey_obj, "ss58_address", None) or str(hotkey_obj)

        if not hasattr(self, "_logged_val_epochs"):
            self._logged_val_epochs: set[int] = set()
        if ep not in self._logged_val_epochs:
            bt.logging.trace(f"[Sybil-PoG] Validator set for epoch {ep} ({len(vals)} entries): {vals}")
            self._logged_val_epochs.add(ep)

        # ─── determine our slot index ─────────────────────────────────────
        if my_hk in vals:
            slots = len(vals)
            idx   = vals.index(my_hk)
        else:
            # not in set → fallback
            if self.allow_fake_sybil_slot:
                bt.logging.warning(f"[Sybil-PoG] Hotkey {my_hk} not in validator set; using fake slot#0 (allow_fake_sybil_slot=True).")
            else:
                bt.logging.warning(
                    f"[Sybil-PoG] Hotkey {my_hk} not in validator set "
                    f"and allow_fake_sybil_slot=False → defaulting to slot#0 in ring size {len(vals)+1}."
                )
            slots = len(vals) + 1
            idx   = 0

        size       = self.blocks_per_epoch // slots
        slot_start = start + idx * size
        slot_end   = start + (idx + 1) * size - 1

        # ─── trace the chosen slot once per epoch ──────────────────────────
        if not hasattr(self, "_logged_slot_epochs"):
            self._logged_slot_epochs: set[int] = set()
        if ep not in self._logged_slot_epochs:
            bt.logging.info(f"[Sybil-PoG] Epoch {ep}: slot {idx}/{slots} → blocks {slot_start}–{slot_end}")
            self._logged_slot_epochs.add(ep)

        return slot_start, slot_end

    def _run_sybil_benchmark(
        self, uid: int, axon: bt.AxonInfo
    ) -> tuple[int, str, bool, str | None, int]:
        """
        Return (uid, hotkey, passed?, gpu_name, num_gpus) for one miner.
        Contains extensive TRACE / INFO logging mirroring validator_sybil.py.
        """
        hotkey = axon.hotkey
        allocation_ok, ssh = False, None

        try:
            bt.logging.trace(f"[Sybil-PoG] ▶ benchmarking {hotkey}")

            # 1) allocation ---------------------------------------------------
            priv, pub = rsa.generate_key_pair()
            bt.logging.trace(f"[Sybil-PoG] {hotkey}: starting allocation")
            fut = asyncio.run_coroutine_threadsafe(
                self.allocate_miner(axon, priv, pub), self.loop
            )
            miner_info = fut.result()
            bt.logging.trace(f"[Sybil-PoG] {hotkey}: miner_info after allocation = {repr(miner_info)}")
            if miner_info is None:
                bt.logging.trace(f"[Sybil-PoG] {hotkey}: allocator busy / no slot (miner_info is None after allocation SUCCESS)")
                return uid, hotkey, False, None, 0

            allocation_ok = True
            bt.logging.trace(f"[Sybil-PoG] {hotkey}: allocated, about to start benchmark, allocation_ok={allocation_ok}")

            # 2) SSH connect --------------------------------------------------
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(miner_info["host"],
                        port     = miner_info.get("port", 22),
                        username = miner_info["username"],
                        password = miner_info["password"],
                        timeout  = 10)
            bt.logging.trace(f"[Sybil-PoG] {hotkey}: SSH OK")

            # 3) upload miner script if needed --------------------------------
            mp_conf  = self.config_data["merkle_proof"]
            m_path   = mp_conf["miner_script_path"]
            miner_py = Path(m_path).read_bytes()

            local_sha  = hashlib.sha256(miner_py).hexdigest()
            try:
                remote_sha = ssh.exec_command(
                    "sha256sum /tmp/miner_script.py | cut -d' ' -f1"
                )[1].read().decode().strip()
            except Exception:
                remote_sha = ""

            if local_sha != remote_sha:
                bt.logging.trace(f"[Sybil-PoG] {hotkey}: uploading miner_script.py")
                with ssh.open_sftp().file("/tmp/miner_script.py", "wb") as f:
                    f.write(miner_py)
                ssh.exec_command("chmod 755 /tmp/miner_script.py")

            # 4) GPU info -----------------------------------------------------
            gpu_info = json.loads(
                ssh.exec_command(
                    "python3 /tmp/miner_script.py --mode gpu_info"
                )[1].read().decode()
            )
            gnum  = gpu_info["num_gpus"]
            gname = gpu_info["gpu_names"][0].strip('" \t')
            bt.logging.trace(f"[Sybil-PoG] {hotkey}: reports {gnum} × {gname}")

            GPU_TM   = self.config_data["gpu_time_models"]
            GPU_VRAM = self.config_data["gpu_performance"]["GPU_AVRAM"]
            BUFFER   = float(mp_conf["buffer_factor"])
            SPOTS    = int(mp_conf["spot_per_gpu"])

            if gname not in GPU_TM or gname not in GPU_VRAM:
                bt.logging.trace(f"[Sybil-PoG] {hotkey}: unknown GPU model")
                return uid, hotkey, False, None, 0

            m       = GPU_TM[gname]
            t_exp   = m["a0"] + m["a1"] * gnum + m["a2"] * (gnum ** 2)
            deadline= t_exp * m["tol"]
            bt.logging.trace(f"[Sybil-PoG] {hotkey}: t_exp={t_exp:.2f}s  "
                            f"deadline={deadline:.2f}s")

            # 5) matrix size --------------------------------------------------
            vram_gib = float(GPU_VRAM[gname])
            n = max(256, int(((vram_gib * BUFFER * 1e9) / (3 * 4)) ** 0.5 // 32 * 32))
            bt.logging.trace(f"[Sybil-PoG] {hotkey}: matrix n={n}")

            # 6) seeds file ---------------------------------------------------
            seeds = {gid: (random.randrange(2**32), random.randrange(2**32))
                    for gid in range(gnum)}
            seed_txt = "\n".join(
                [str(n)] + [f"{gid} {a} {b}" for gid, (a, b) in seeds.items()]
            )
            with ssh.open_sftp().file("/tmp/seeds.txt", "wb") as f:
                f.write(seed_txt.encode())

            # 7) compute phase ------------------------------------------------
            t0   = time.time()
            out  = ssh.exec_command(
                "python3 /tmp/miner_script.py --mode compute"
            )[1].read().decode()
            t_compute = time.time() - t0
            bt.logging.trace(f"[Sybil-PoG] {hotkey}: compute {t_compute:.2f}s")
            if t_compute > deadline:
                bt.logging.trace(f"[Sybil-PoG] {hotkey}: exceeded deadline")
                return uid, hotkey, False, None, 0

            roots = {gid: bytes.fromhex(rh) for gid, rh in
                    json.loads(next(l[6:] for l in out.splitlines()
                                    if l.startswith("ROOTS:")))}

            # 8) challenge indices -------------------------------------------
            idxs = {
                gid: [(random.randrange(0, 2 * n), random.randrange(0, n))
                    for _ in range(SPOTS)]
                for gid in range(gnum)
            }
            idx_txt = "\n".join(
                f"{gid} " + ";".join(f"{i},{j}" for i, j in pairs)
                for gid, pairs in idxs.items()
            )
            with ssh.open_sftp().file("/tmp/challenge_indices.txt", "wb") as f:
                f.write(idx_txt.encode())

            # 9) proof phase --------------------------------------------------
            ssh.exec_command("python3 /tmp/miner_script.py --mode proof")[1].read()

            # 10) download responses -----------------------------------------
            resps = {}
            with ssh.open_sftp() as sftp:
                for gid in range(gnum):
                    tmp = tempfile.NamedTemporaryFile(delete=False).name
                    sftp.get(f"/dev/shm/resp_{gid}.npy", tmp)
                    resps[gid] = np.load(tmp, allow_pickle=True).item()
                    os.unlink(tmp)

            # 11) verification ----------------------------------------------
            for gid in range(gnum):
                sA, sB = seeds[gid]
                for (i, j), row, proof in zip(
                        idxs[gid], resps[gid]["rows"], resps[gid]["proofs"]):

                    if i < n:
                        exp = sum(prng(sA, i, k) * prng(sB, k, j) for k in range(n))
                    else:
                        exp = sum(prng(sB, i - n, k) * prng(sA, k, j) for k in range(n))

                    if (not np.isclose(exp, row[j], rtol=1e-3, atol=1e-4) or
                            not merkle_ok(row, proof, roots[gid], i, 2 * n)):
                        bt.logging.trace(f"[Sybil-PoG] {hotkey}: proof mismatch")
                        return uid, hotkey, False, None, 0

            bt.logging.trace(f"[Sybil-PoG] {hotkey}: PASS")
            return uid, hotkey, True, gname, gnum

        except Exception as e:
            bt.logging.trace(f"[Sybil-PoG] {hotkey}: exception {e}")
            return uid, hotkey, False, None, 0

        finally:
            if allocation_ok:
                asyncio.run_coroutine_threadsafe(
                    self.deallocate_miner(axon, pub), self.loop
                )
            if ssh:
                try:
                    ssh.close()
                except Exception:
                    pass

    async def proof_of_gpu_sybil(self) -> None:
        """
        Orchestrates the Sybil-PoG benchmark **once per Sybil epoch** inside
        our personal validator slot.

        High-level flow:
        • wait until our slot starts (w/ small random jitter)
        • pick miners that are *not* currently allocated elsewhere
        • run `_run_sybil_benchmark()` on them in parallel
        • update / purge PoG-DB entries accordingly
        """
        try:
            # ─── 1 determine slot ───────────────────────────────────────────
            slot_start, slot_end = self.my_sybil_slot()
            bt.logging.trace(
                f"[Sybil-PoG] My slot this epoch: blocks {slot_start}–{slot_end}"
            )

            latest_start = slot_end - self.max_challenge_blocks
            if self.current_block >= latest_start:
                bt.logging.debug("[Sybil-PoG] Slot almost over – skipping.")
                return

            # ─── 2 un-predictable delay so miners can’t pre-compute ─────────
            delay_blocks = random.randint(
                1, min(self.rand_delay_blocks_max, latest_start - self.current_block),
            )
            bt.logging.trace(
                f"[Sybil-PoG] Sleeping {delay_blocks} blocks "
                f"({delay_blocks*12}s) before launching challenges."
            )
            await asyncio.sleep(delay_blocks * 12)

            # ─── 3 choose target miners ─────────────────────────────────────
            self._queryable_uids = self.get_queryable()
            allocated = self.wandb.get_allocated_hotkeys(
                self.get_valid_validator_hotkeys(), True
            )
            axons = [
                (uid, ax) for uid, ax in self._queryable_uids.items()
                if ax.hotkey not in allocated
            ]
            bt.logging.info(f"[Sybil-PoG] Challenging {len(axons)} miners")

            # ─── 4 launch benchmarks in parallel threads ────────────────────
            loop = asyncio.get_running_loop()

            num_miners = len(axons)
            bt.logging.trace(f"[Sybil-PoG] Launching benchmarks for {num_miners} miners with max_workers={num_miners}")

            # dedicate a *fresh* pool large enough for *all* miners this round
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(axons)) as pool:
                tasks = [
                    loop.run_in_executor(pool, self._run_sybil_benchmark, uid, ax)
                    for uid, ax in axons
                ]
                slot_size = slot_end - slot_start
                available_blocks = max(0, slot_size - delay_blocks)
                timeout_blocks = min(self.max_challenge_blocks, available_blocks)
                timeout_seconds = max(60, timeout_blocks * 12)

                bt.logging.trace(f"[Sybil-PoG] Timeout set to {timeout_seconds:.1f}s "
                                f"(slot size: {slot_size}, delay: {delay_blocks}, "
                                f"available (blocks): {available_blocks}, capped to (blocks): {timeout_blocks})")

                try:
                    done, pending = await asyncio.wait(tasks, timeout=timeout_seconds)
                    bt.logging.info(f"[Sybil-PoG] Benchmarks done. {len(done)} completed, {len(pending)} pending")
                except Exception as e:
                    bt.logging.error(f"[Sybil-PoG] Exception during asyncio.wait: {e}")
                    return

            for fut in pending:
                fut.cancel()
                bt.logging.warning(f"[Sybil-PoG] Task {fut} timed out and was cancelled")

            for fut in done:
                try:
                    result = fut.result()
                    # Log result
                except Exception as e:
                    bt.logging.error(f"[Sybil-PoG] Task Exception: {e!r}")

            # ─── 5 post-process results ─────────────────────────────────────
            passed, failed = 0, 0

            # 1) all on-chain hotkeys
            onchain_hotkeys = {
                self.metagraph.axons[uid].hotkey
                for uid in self.uids
            }

            # 2) track who actually passed this round
            passed_hotkeys = set()
            for fut in done:
                uid, hotkey, ok, gname, gnum = fut.result()
                if ok:
                    update_pog_stats(self.db, hotkey, gname, gnum)
                    passed += 1
                    passed_hotkeys.add(hotkey)
                else:
                    failed += 1

            # 3) rebuild penalized_hotkeys from scratch:
            #    everyone on-chain who didn’t pass this round
            self.penalized_hotkeys = [
                hk for hk in onchain_hotkeys
                if hk not in passed_hotkeys
            ]

            bt.logging.trace(f"PoG post-process: {passed} passed, {failed} failed; penalized: {self.penalized_hotkeys}")

            bt.logging.success(
                f"[Sybil-PoG] Completed – {passed} PASS, {failed} FAIL, "
                f"{len(pending)} timeout."
            )
            self.sync_scores()

        except Exception as e:
            bt.logging.error(f"[Sybil-PoG] exception: {e}\n{traceback.format_exc()}")

    async def start(self):
        """The Main Validation Loop"""
        self.loop = asyncio.get_running_loop()

        # Step 5: Perform queries to miners, scoring, and weight
        block_next_pog = 1
        block_next_sync_status = 1
        block_next_set_weights = self.current_block + weights_rate_limit
        block_next_hardware_info = 1
        block_next_miner_checking = 1

        last_pog_epoch   = -1
        last_sybil_epoch = -1

        time_next_pog = None
        time_next_sync_status = None
        time_next_set_weights = None
        time_next_hardware_info = None

        time_next_legacy_pog  = None
        time_next_sybil       = None
        block_next_legacy_pog = 1
        block_next_sybil      = 1

        bt.logging.info("Starting validator loop.")
        while True:
            try:
                self.sync_local()
                epoch = self.current_epoch()
                self.refresh_config_from_server()

                hk_obj = self.wallet.hotkey
                my_hk  = getattr(hk_obj, "ss58_address", None) or str(hk_obj)

                if self.current_block not in self.blocks_done:
                    self.blocks_done.add(self.current_block)

                    time_next_pog = self.next_info(not block_next_pog == 1, block_next_pog)
                    time_next_sync_status = self.next_info(not block_next_sync_status == 1, block_next_sync_status)
                    time_next_set_weights = self.next_info(not block_next_set_weights == 1, block_next_set_weights)
                    time_next_hardware_info = self.next_info(
                        not block_next_hardware_info == 1 and self.validator_perform_hardware_query, block_next_hardware_info
                    )

                    if self.epoch_is_pog(epoch):          # we are in an even epoch already
                        next_legacy_epoch = epoch + 2     # jump to the *next* even epoch
                    else:                                 # we are in an odd epoch
                        next_legacy_epoch = epoch + 1     # next epoch is even
                    block_next_legacy_pog = self.epoch_start_block(next_legacy_epoch)
                    time_next_legacy_pog  = calculate_next_block_time(
                        self.current_block, block_next_legacy_pog
                    )

                    # Next personal Sybil slot
                    if not self.epoch_is_pog(epoch):        # we’re already in a Sybil epoch
                        slot_start, _ = self.my_sybil_slot(epoch)
                        if self.current_block < slot_start:         # slot still ahead
                            block_next_sybil = slot_start
                            next_sybil_epoch = epoch
                        else:                                       # slot has passed: jump 2 epochs
                            next_sybil_epoch = epoch + 2
                            block_next_sybil, _ = self.my_sybil_slot(next_sybil_epoch)
                    else:                                   # currently even epoch → next epoch is odd
                        next_sybil_epoch = epoch + 1
                        block_next_sybil, _ = self.my_sybil_slot(next_sybil_epoch)

                    time_next_sybil = calculate_next_block_time(
                        self.current_block, block_next_sybil
                    )

                    DEBUG_FAST_SYBIL = False   # <— flip to False to restore normal scheduling

                    if DEBUG_FAST_SYBIL:
                        # one-time initialisation
                        if not hasattr(self, "_next_fast_sybil_block"):
                            self._next_fast_sybil_block = self.current_block + 	self.max_challenge_blocks + 1
                            bt.logging.info(
                                f"[DEBUG] First fast Sybil planned at block {self._next_fast_sybil_block}"
                            )

                        # launch when due
                        if self.current_block >= self._next_fast_sybil_block:
                            if self.gpu_task is None or self.gpu_task.done():
                                bt.logging.info("[DEBUG] Launching fast-loop proof_of_gpu_sybil()")
                                # skip the slot checks INSIDE that function for debug runs
                                self.gpu_task = asyncio.create_task(self.proof_of_gpu_sybil())

                            # schedule the next run just once, *after* we decide to launch
                            jitter  = random.randint(1, self.rand_delay_blocks_max)   # 1-5 blocks
                            cushion = random.randint(2, 3)                       # safety margin
                            self._next_fast_sybil_block = (
                                self.current_block + self.max_challenge_blocks + jitter + cushion
                            )
                            bt.logging.info(
                                f"[DEBUG] Next fast Sybil planned at block {self._next_fast_sybil_block}"
                            )

                    else:
                        if self.epoch_is_pog(epoch):           # even → legacy PoG
                            if epoch != last_pog_epoch and \
                            self.current_block % self.blocks_per_epoch == 0:
                                if self.gpu_task is None or self.gpu_task.done():
                                    self.gpu_task = asyncio.create_task(self.proof_of_gpu())
                                    self.gpu_task.add_done_callback(self.on_gpu_task_done)
                                last_pog_epoch = epoch
                        else:                                   # odd → PoG-Sybil
                            if my_hk in self.sybil_eligible_hotkeys:
                                slot_start, _ = self.my_sybil_slot(epoch)
                                if (epoch != last_sybil_epoch and
                                        self.current_block == slot_start):
                                    if self.gpu_task is None or self.gpu_task.done():
                                        self.gpu_task = asyncio.create_task(self.proof_of_gpu_sybil())
                                    last_sybil_epoch = epoch

                    # Perform specs queries
                    if (self.current_block % block_next_hardware_info == 0 and self.validator_perform_hardware_query) or (
                        block_next_hardware_info < self.current_block and self.validator_perform_hardware_query
                    ):
                        block_next_hardware_info = self.current_block + 150  # 150 -> ~ every 30 minutes

                        if not hasattr(self, "_queryable_uids"):
                            self._queryable_uids = self.get_queryable()

                        # self.loop.run_in_executor(None, self.execute_specs_request) replaced by wandb query.
                        await self.get_specs_wandb()

                    # Perform miner checking
                    if self.current_block % block_next_miner_checking == 0 or block_next_miner_checking < self.current_block:
                        # Next block the validators will do port checking again.
                        block_next_miner_checking = self.current_block + 50  # 300 -> every 60 minutes

                        # Filter axons with stake and ip address.
                        self._queryable_uids = self.get_queryable()

                        # self.sync_checklist()

                    if self.current_block % block_next_sync_status == 0 or block_next_sync_status < self.current_block:
                        block_next_sync_status = self.current_block + 25  # ~ every 5 minutes
                        self.sync_status()
                        # Log chain data to wandb
                        chain_data = {
                            "Block": self.current_block,
                            "Stake": float(self.metagraph.S[self.validator_subnet_uid]),
                            "Rank": float(self.metagraph.R[self.validator_subnet_uid]),
                            "vTrust": float(self.metagraph.validator_trust[self.validator_subnet_uid]),
                            "Emission": float(self.metagraph.E[self.validator_subnet_uid]),
                        }
                        self.wandb.log_chain_data(chain_data)

                    # Periodically update the weights on the Bittensor blockchain, ~ every 20 minutes
                    if self.current_block - self.last_updated_block > weights_rate_limit:
                        block_next_set_weights = self.current_block + weights_rate_limit
                        self.sync_scores()
                        self.set_weight_capped_by_gpu()
                        self.last_updated_block = self.current_block
                        self.blocks_done.clear()
                        self.blocks_done.add(self.current_block)

                    # Refresh tokens periodically (every 30 minutes)
                    if self.current_block % 600 == 0:  # Approximately every 30 minutes at 3s block time
                        bt.logging.info("Refreshing SN27 token gateway tokens")
                        self.pubsub_client.refresh_credentials()

                await asyncio.sleep(1)

            # If we encounter an unexpected error, log it for debugging.
            except RuntimeError as e:
                bt.logging.error(e)
                traceback.print_exc()

            # If the user interrupts the program, gracefully exit.
            except KeyboardInterrupt:
                self.db.close()
                bt.logging.success("Keyboard interrupt detected. Exiting validator.")
                exit()

            bt.logging.info(
            (
                f"Block:{self.current_block} | "
                f"Stake:{self.metagraph.S[self.validator_subnet_uid]} | "
                f"Rank:{self.metagraph.R[self.validator_subnet_uid]} | "
                f"vTrust:{self.metagraph.validator_trust[self.validator_subnet_uid]} | "
                f"Emission:{self.metagraph.E[self.validator_subnet_uid]} | "
                f"next_PoG_legacy:  #{block_next_legacy_pog} ~ {time_next_legacy_pog} | "
                f"next_PoG_sybil: #{block_next_sybil} ~ {time_next_sybil} | "
                f"sync_status:   #{block_next_sync_status} ~ {time_next_sync_status} | "
                f"set_weights:   #{block_next_set_weights} ~ {time_next_set_weights} | "
            )
)

def main():
    """
    Main function to run the neuron.

    This function initializes and runs the neuron. It handles the main loop, state management, and interaction
    with the Bittensor network.
    """
    validator = Validator()
    asyncio.run(validator.start())
    asyncio.run(validator.pubsub_client.subscribe_to_topics())


if __name__ == "__main__":
    main()
