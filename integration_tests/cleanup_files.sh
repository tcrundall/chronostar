# remove subdirectories
find . -mindepth 2 -maxdepth 2 -type d -exec rm -r '{}' \;
find . -mindepth 2 ! -name '*.pyc' ! -name '*.py' ! -name '*.txt' -exec rm '{}' \;
