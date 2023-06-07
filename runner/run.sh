#!/bin/bash
cd /home/runtime/runner && python parse_config.py && \
	helics run --path /home/run/test_system_runner.json
