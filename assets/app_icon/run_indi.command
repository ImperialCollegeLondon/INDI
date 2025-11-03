#!/bin/bash

open -a "Terminal"

echo " "
echo "================== YAML settings settings =================="
read yaml_path

echo " "
echo "==================     CDTI scan path     =================="
read cdti_path

echo " "
echo "Running INDI... "

cd ~/GitHub/INDI
~/Github/INDI/.venv/bin/indi $yaml_path --start_folder $cdti_path