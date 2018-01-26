# Trains a series of different models and outputs them to one file


WORD_VECTOR_FILE=/home/eric/Documents/word_vectors/GloVe/glove.vec
TRAIN_FILE=dataset/preprocessed_train.csv
TEST_FILE=dataset/preprocessed_test.csv
SEQ_LENGTH=150
ENSEMBLE_DIR=enesmbles/128_glove_6b_ensemble_lda_avg
LDA_DIM=25
NUMBER_OF_EPOCHS=10

python3 train_basic.py --train $TRAIN_FILE --test $TEST_FILE \
--model birnn --seq-length $SEQ_LENGTH --word2vec $WORD_VECTOR_FILE \
--experiment-name birnn_300_glove_6b_ensemble_lda_avg \
--ensemble-dir $ENSEMBLE_DIR \
--epoch $NUMBER_OF_EPOCHS --use-lda 1 --lda-dim $LDA_DIM

python3 train_basic.py --train $TRAIN_FILE --test $TEST_FILE \
--model clstm --seq-length $SEQ_LENGTH --word2vec $WORD_VECTOR_FILE  \
--experiment-name clstm_300_glove_6b_ensemble_lda_avg \
--ensemble-dir $ENSEMBLE_DIR \
--epoch $NUMBER_OF_EPOCHS --use-lda 1 --lda-dim $LDA_DIM

# python3 train_basic.py --train $TRAIN_FILE --test $TEST_FILE \
# --model dpcnn --seq-length $SEQ_LENGTH --word2vec $WORD_VECTOR_FILE \
# --experiment-name dpcnn_300_glove_6b_ensemble_ldaa \
# --ensemble-dir $ENSEMBLE_DIR \
# --epoch $NUMBER_OF_EPOCHS --use-lda 1 --lda-dim $LDA_DIM
#
# python3 train_basic.py --train $TRAIN_FILE --test $TEST_FILE \
# --model birnn --seq-length $SEQ_LENGTH --word2vec $WORD_VECTOR_FILE \
# --experiment-name birnn_300_glove_6b_ensemble_no_lda \
# --ensemble-dir $ENSEMBLE_DIR \
# --epoch $NUMBER_OF_EPOCHS --use-lda 0 --lda-dim $LDA_DIM
#
# python3 train_basic.py --train $TRAIN_FILE --test $TEST_FILE \
# --model clstm --seq-length $SEQ_LENGTH --word2vec $WORD_VECTOR_FILE  \
# --experiment-name clstm_300_glove_6b_ensemble_no_lda \
# --ensemble-dir $ENSEMBLE_DIR \
# --epoch $NUMBER_OF_EPOCHS --use-lda 0 --lda-dim $LDA_DIM
#
# python3 train_basic.py --train $TRAIN_FILE --test $TEST_FILE \
# --model dpcnn --seq-length $SEQ_LENGTH --word2vec $WORD_VECTOR_FILE \
# --experiment-name dpcnn_300_glove_6b_ensemble_no_lda \
# --ensemble-dir $ENSEMBLE_DIR \
# --epoch $NUMBER_OF_EPOCHS --use-lda 0 --lda-dim $LDA_DIM
