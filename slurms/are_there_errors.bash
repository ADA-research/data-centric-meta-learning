#!/bin/bash
for file in *.err; do
  if [ -s $file ]; then
    echo "$file is not empty."
  fi
done