#!/bin/bash
cd /home/runtime/runner && python parse_config.py && \
	helics run --path /home/run/system_runner.json