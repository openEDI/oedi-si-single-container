#!/bin/bash
rm /home/app/logs/*.log
python3 /home/app/runner/parse_config.py && \
helics -v run --path=/home/app/runner/config_runner.json && \
mv *.log /home/app/logs
