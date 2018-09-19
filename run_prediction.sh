TEST_PATH=/your/path/to/folder/containing/images/
OUTPUT_PATH=/your/path/to/folder/for/preductions/
MODEL_PATH=baseline-resnet50_dilated8-ppm_bilinear_deepsup
RESULT_PATH=./

ENCODER=$MODEL_PATH/encoder_epoch_20.pth
DECODER=$MODEL_PATH/decoder_epoch_20.pth

if [ ! -e $ENCODER ]; then
  mkdir $MODEL_PATH
fi
if [ ! -e $ENCODER ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$ENCODER
fi
if [ ! -e $DECODER ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$DECODER
fi
if [ ! -e $TEST_PATH ]; then
  wget -P $RESULT_PATH http://sceneparsing.csail.mit.edu//data/ADEChallengeData2016/images/validation/$TEST_PATH
fi

python -u test.py \
  --model_path $MODEL_PATH \
  --output_path $OUTPUT_PATH \
  --test_img $TEST_PATH \
  --arch_encoder resnet50_dilated8 \
  --arch_decoder ppm_bilinear_deepsup \
  --fc_dim 2048
