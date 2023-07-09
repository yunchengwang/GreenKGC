DATASET=$1
BASELINE=$2
FEATURE="$BASELINE"_pruned
NEG=$3
LCW=$4

echo $FEATURE

python -u feature_pruning.py $DATASET $BASELINE 512

python decision.py --dataset $DATASET \
                   --pretrained_emb $FEATURE \
                   --dim 32 -o output \
                   --negative_size 256 \
                   --max_depth 5 \
                   --n_estimators 2000 \
                   --neg_sampling $NEG \
                   --lcwa --lcw_threshold $LCW

python decision.py --dataset $DATASET \
                   --pretrained_emb $FEATURE \
                   --dim 100 -o output \
                   --negative_size 256 \
                   --max_depth 5 \
                   --n_estimators 2000 \
                   --neg_sampling $NEG \
                   --lcwa --lcw_threshold $LCW
