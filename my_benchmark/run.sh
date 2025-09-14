nsys profile \
  --stats=true \
  -t cuda,nvtx,cublas,cudnn,osrt \
  --show-output=true \
  -f true \
  -o output/fiddler \
  --sample=none \
  python ../src/fiddler/infer.py