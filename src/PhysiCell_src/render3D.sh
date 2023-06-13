#!/bin/bash

# Use the find command to locate files matching the pattern "snapshot*.svg"
# in the directory ./output and its subdirectories, and pipe the results to
# the wc command to count the number of matching files.
count=$(find ./output -name 'snapshot*.svg' | wc -l)

# Subtract 1 from the count.
count=$((count - 1))

# Print the number of matching files minus 1.
echo "There are $count files matching the pattern 'snapshot*.svg' in the directory ./output, excluding the current script."

./povwriter 0:1:$count

for scene in *.pov; do povray +w640 +h480 +a0.3 -D $scene; done

