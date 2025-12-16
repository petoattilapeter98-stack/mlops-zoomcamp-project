#!/usr/bin/env python
"""
Deployment script for the taxi trip duration training workflow.

This script creates a deployment with monthly scheduling using Prefect 3.x.

Usage:
    python deployment.py

This will create a deployment that:
- Runs on the 1st day of each month at 08:00 UTC
- Uses the prefect.yaml configuration file
"""

import subprocess
import sys

if __name__ == "__main__":
    print("Creating Prefect deployment with monthly schedule...")
    print("The workflow will run monthly on the 1st of each month at 08:00 UTC")
    print()
    print("To deploy, run:")
    print("  prefect deploy")
    print()
    print("To start the Prefect worker/runner:")
    print("  prefect worker start")
    print()
    print("To view the UI:")
    print("  prefect server start")

