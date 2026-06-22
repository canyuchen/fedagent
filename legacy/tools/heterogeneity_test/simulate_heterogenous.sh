python "$(dirname "$0")/simulate_preference.py" --dataset webshop

python "$(dirname "$0")/simulate_preference.py" --dataset alfworld


python "$(dirname "$0")/simulate_coverage.py" --dataset webshop
python "$(dirname "$0")/simulate_coverage.py" --dataset alfworld

python "$(dirname "$0")/simulate_hardness.py" --dataset webshop
python "$(dirname "$0")/simulate_hardness.py" --dataset alfworld