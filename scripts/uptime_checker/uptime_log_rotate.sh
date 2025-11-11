#!/bin/bash
SCRIPT_PATH="/home/sn27/uptime_checker"

cp $SCRIPT_PATH/uptime.log.2 $SCRIPT_PATH/uptime.log.3
cp $SCRIPT_PATH/uptime.log.1 $SCRIPT_PATH/uptime.log.2
cp $SCRIPT_PATH/uptime.log $SCRIPT_PATH/uptime.log.1
:>$SCRIPT_PATH/uptime.log
