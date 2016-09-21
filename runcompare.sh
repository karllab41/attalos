#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: runcompare.sh <filename-prefix>"
fi

numcount=20

echo $1-fixed.txt
tail -$numcount ${1}fixed.txt
echo $1-optim.txt
tail -$numcount ${1}optim.txt

