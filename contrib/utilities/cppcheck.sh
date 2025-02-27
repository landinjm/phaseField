#!/bin/bash

#
# This script runs cppcheck on the PRISMS-PF code base.
#
#
# Usage:
# ./contrib/utilities/cppcheck.sh
#

if test ! -d src -o ! -d include -o ! -d applications ; then
  echo "This script must be run from the top-level directory of PRISMS-PF"
  exit 0
fi

if ! [ -x "$(command -v cppcheck)" ] ; then
    echo "make sure cppcheck is in the path"
    exit 1
fi

# Directories containing the .cc and .h files
TARGET_DIRS=("src" "include" "tests" "applications")

for TARGET_DIR in "${TARGET_DIRS[@]}" ; do
  # Check if the directory exists
  if [ ! -d "$TARGET_DIR" ] ; then
    echo "Directory $TARGET_DIR does not exist"
    exit 2
  fi

  # Find all .cc and .h files and run cppcheck on each
  find "$TARGET_DIR" -type f \( -name "*.cc" -o -name "*.h" \) -print0 | while IFS= read -r -d '' FILE ; do
    cppcheck --enable=all --language=c++ --std=c++17 --suppress=missingIncludeSystem --suppress=unknownMacro "$FILE" >> "output.txt" 2>&1
  done
done

# grep interesting errors and make sure we remove duplicates:
grep -E '(warning|error|style): ' output.txt | sort | uniq >cppcheck.log

# If we have errors, report them and set exit status to failure
if [ -s cppcheck.log ]; then
    cat cppcheck.log
    exit 4
fi

echo "OK"
exit 0