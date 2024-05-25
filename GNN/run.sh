# Set default values for arguments
DEFAULT_ALPHAS="0.000000,0.523599,1.047200,1.570800,x.xxxxxx"
DEFAULT_FRAC=0.5
DEFAULT_FILE_DIR="./files/"

# Check for mandatory argument (train or test)
if [ -z "$1" ]; then
  echo "Error: Please provide a mode (train or test) as the first argument."
  exit 1
fi

# Check if the mode is valid
if [[ ! "$1" =~ ^(train|test)$ ]]; then
  echo "Error: Invalid mode. Please use 'train' or 'test'."
  exit 1
fi

# Get arguments
MODE="$1"
FILE_DIR=${2:-$DEFAULT_FILE_DIR}  # Use default for FILE_DIR if empty
ALPHAS=${3:-$DEFAULT_ALPHAS}    # Use default if empty
FRAC=${4:-$DEFAULT_FRAC}        # Use default if empty

# Process remaining arguments (excluding first 4)
declare -a REMAINING_ARGS        # Declare an array for remaining arguments

shift 4                            # Shift first 4 arguments

# Loop through remaining arguments and store them in the array
while [[ -n "$1" ]]; do
  REMAINING_ARGS+=("$1")
  shift
done

# Check for required arguments in test mode
if [ "$MODE" == "test" ]; then
  if [ -z "${REMAINING_ARGS[1]}" ]; then
    echo "Error: Checkpoint path is required for testing."
    exit 1
  fi
  if [ -z "${REMAINING_ARGS[2]}" ]; then
    echo "Error: Data name is required for testing."
    exit 1
  fi
fi

# Run commands
if [ "$MODE" == "train" ]; then
  echo "Running graph creation..."
  ipython data/create_graphs.py -- --file_dir "$FILE_DIR" --alphas "$ALPHAS" --frac "$FRAC"
  echo "Running training script..."
  ipython main.py -- fit --config configs/config.yaml
else
  echo "Running $MODE..."
  echo ipython main.py -- test --config configs/config.yaml \
    --ckpt_path "${REMAINING_ARGS[1]}" --data.name "${REMAINING_ARGS[2]}"  # Use array access for test arguments
  ipython main.py -- test --config configs/config.yaml \
    --ckpt_path "${REMAINING_ARGS[1]}" --data.name "${REMAINING_ARGS[2]}"  # Use array access for test arguments
    # You can access additional arguments from the REMAINING_ARGS array if needed
fi
