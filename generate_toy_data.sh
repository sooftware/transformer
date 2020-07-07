#! /bin/sh

# N.B.: assumes script is called from parent directory, as described in README.md
# shellcheck disable=SC2164
cd toy_problem
python generate_toy_data.py

# shellcheck disable=SC2103
cd ..