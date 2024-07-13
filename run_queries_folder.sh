#!/bin/bash

# Directory to loop through
DIRECTORY=$1
RES_DIR="RESULTS/$DIRECTORY"
mkdir -p $RES_DIR
TIMEOUT=300
# Loop through all files in the directory
for FILE in "$DIRECTORY"/*.txt
do
  # Check if it is a file (not a directory)
  if [ -f "$FILE" ]; then
    ABSOLUTE_FILE=$(realpath "$FILE")
    BASE_FILE=$(basename "$FILE")
    echo "Solving $ABSOLUTE_FILE"
    ~/opt/Marabou/build/bin/Marabou --input-query $ABSOLUTE_FILE --timeout $TIMEOUT > $RES_DIR/$BASE_FILE.output
    
  fi
done