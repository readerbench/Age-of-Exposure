#! /bin/sh

QUANTILES_FILE=$1
EXPERIMENT_NAME=$2
VECTOR_SIZE="${3:-300}"


echo "Running experiment *$EXPERIMENT_NAME* on *$QUANTILES_FILE* with "\
     "vector size *$VECTOR_SIZE*"


MODEL_DIR="models/$EXPERIMENT_NAME"
INDICES_FILE="indices/indices_$EXPERIMENT_NAME.csv"
DISPARITIES_FILE="indices/disparity_$EXPERIMENT_NAME.json"
TF_FILE="$MODEL_DIR/term_frequency.json"
PREDICTIONS_FILE="aoe_v2_predictions.csv"


echo "\tindices file: $INDICES_FILE"
echo "\tdisparities file: $DISPARITIES_FILE"
echo "\tmodel dir: $MODEL_DIR"

python train_models.py -i $QUANTILES_FILE -s $VECTOR_SIZE -o $MODEL_DIR
python compute_indices.py -i $MODEL_DIR -d $DISPARITIES_FILE -o $INDICES_FILE 
python evaluate_aoa.py -i $INDICES_FILE -o $EXPERIMENT_NAME -tf $TF_FILE
python generate_aoe_v2_predictions.py -i $INDICES_FILE -tf $TF_FILE -o $PREDICTIONS_FILE