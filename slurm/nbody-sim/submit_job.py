#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

"""
Script to submit N-body simulation jobs to Slurm.
"""

import os
import sys
import argparse
import subprocess
import json
import time
from datetime import datetime


def submit_job(args):
    """
    Submit a job to Slurm with the given parameters.
    
    Args:
        args: Command-line arguments
    
    Returns:
        Job ID if successful, None otherwise
    """
    # Build sbatch command
    command = [
        "sbatch",
        "--parsable",  # Output just the job ID
    ]
    
    # Add partition if specified
    if args.partition:
        command.extend(["--partition", args.partition])
    
    # Add node constraint if specified
    if args.nodes:
        nodes_list = ",".join(args.nodes)
        command.extend(["--nodelist", nodes_list])
    
    # Add job script and parameters
    command.extend([
        os.path.join(os.path.dirname(__file__), "nbody_job.sbatch"),
        args.system_type,
        str(args.num_particles),
        str(args.duration),
        str(args.time_step),
        args.integrator
    ])
    
    # Print the command
    print(f"Submitting job with command: {' '.join(command)}")
    
    # Submit the job
    try:
        job_id = subprocess.check_output(command).decode().strip()
        print(f"Job submitted successfully with ID: {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}")
        return None


def monitor_job(job_id, wait=False):
    """
    Monitor a Slurm job.
    
    Args:
        job_id: Slurm job ID
        wait: Whether to wait for job completion
    """
    if not job_id:
        return
    
    while True:
        # Get job information
        try:
            cmd = ["scontrol", "show", "job", job_id]
            output = subprocess.check_output(cmd).decode()
            
            # Parse job state
            for line in output.split("\n"):
                if "JobState=" in line:
                    state = line.split("JobState=")[1].split()[0]
                    break
            else:
                state = "UNKNOWN"
            
            print(f"Job {job_id} is in state: {state}")
            
            # If job is no longer running and we're waiting, exit
            if state not in ["PENDING", "RUNNING", "CONFIGURING"] and wait:
                print(f"Job {job_id} has completed with state: {state}")
                break
            
        except subprocess.CalledProcessError:
            print(f"Error getting information for job {job_id}")
            if wait:
                # If we can't get job info and we're waiting, assume it's done
                break
            else:
                return
        
        # If not waiting, just show status once
        if not wait:
            break
        
        # Wait before checking again
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description="Submit N-body simulation to Slurm")
    
    # Slurm parameters
    parser.add_argument("--partition", type=str, default=None,
                        help="Slurm partition to use")
    parser.add_argument("--nodes", type=str, nargs="+", default=None,
                        help="Specific nodes to use (e.g., 'orin1 orin2')")
    parser.add_argument("--wait", action="store_true",
                        help="Wait for job to complete")
    
    # Simulation parameters
    parser.add_argument("--system-type", choices=["random", "solar", "galaxy"],
                        default="galaxy", help="Type of system to simulate")
    parser.add_argument("--num-particles", type=int, default=5000,
                        help="Number of particles to simulate")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Simulation duration in simulation time units")
    parser.add_argument("--time-step", type=float, default=0.01,
                        help="Simulation time step")
    parser.add_argument("--integrator", choices=["euler", "leapfrog", "verlet", "rk4"],
                        default="leapfrog", help="Integration method")
    
    args = parser.parse_args()
    
    # Submit the job
    job_id = submit_job(args)
    
    # Monitor the job if requested
    if job_id and args.wait:
        monitor_job(job_id, wait=True)


if __name__ == "__main__":
    main()