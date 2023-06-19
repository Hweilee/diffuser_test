###controlnet.sh PSEUDO###
env_name=${1}
model_name=${2}
batch_size=${3}
output_dir=${4:-"outputs"}
current_dir=$(pwd)
log_dir=${5:-"${current_dir}/logs/${model_name}.json"}

# Check if $batch_size is provided , if not use default value of 5
if expr "$batch_size" + 0 > /dev/null 2>&1; then
  batch_size=$batch_size
else
  batch_size=5
fi

moreh-switch-model -M 1 

# Run training script
echo "# ========================================================= #"
echo "training ${model_name}.."
conda run -n ${env_name} python3 train_controlnet.py \
    --pretrained_model_name_or_path ${model_name}  \
    --controlnet_model_name_or_path lllyasviel/sd-controlnet-hed \
    --dataset_name fusing/fill50k \
    --output_dir ${output_dir} \
    --log_dir "${log_dir}" \
    --learning_rate=1e-5 \
    --max_train_steps 10 \
    --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
    --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
    --train_batch_size ${batch_size} \