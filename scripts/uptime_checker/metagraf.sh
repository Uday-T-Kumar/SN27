#!/bin/bash

# Path to virtual environment
VENV_PATH="/home/sn27/SN27/venv"

# Path of Python script
SCRIPT_PATH="/home/sn27/uptime_checker"

# Activate the venv
source "$VENV_PATH/bin/activate"

#Take a backup in case there is an error during script run
cp $SCRIPT_PATH/metagrafdata.txt.bkp $SCRIPT_PATH/metagrafdata.txt.bkp1
cp $SCRIPT_PATH/metagrafdata.txt $SCRIPT_PATH/metagrafdata.txt.bkp

#truncate the txt path
:>$SCRIPT_PATH/metagrafdata.txt

# Run the Python script
python3 "$SCRIPT_PATH/metagraf.py"

# Check exit status
if [ $? -eq 0 ]; then
        echo `date`" :Execution successful"
else
        echo `date`" :Execution failed"
        cp $SCRIPT_PATH/metagrafdata.txt.bkp $SCRIPT_PATH/metagrafdata.txt
fi

SIZE=`cat metagrafdata.txt | wc -l`
echo `date`" -size of metagraphdata.txt: " $SIZE
if [ $SIZE -eq 0 ]; then
        cp $SCRIPT_PATH/metagrafdata.txt.bkp $SCRIPT_PATH/metagrafdata.txt
fi

# Deactivate the venv
deactivate
