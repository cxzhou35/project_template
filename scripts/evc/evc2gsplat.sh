START_FRAMES=200
END_FRAMES=259
DATA_ROOT="/home/zhouchenxu/datasets/vsr_4dv/shijie_far"
ORIG_OUT_DIR="/home/zhouchenxu/codes/gsplat/data/shijie_far/orig"
VSR_OUT_DIR="/home/zhouchenxu/codes/gsplat/data/shijie_far/vsr_crop"

# echo "=== Running EVC to COLMAP for orig data ==="
# python3 scripts/evc/evc_to_colmap_full.py \
#     --data_root $DATA_ROOT \
#     --output $ORIG_OUT_DIR \
#     --image_path images \
#     --image_ext .jpg \
#     --intri_path intri.yml \
#     --extri_path extri.yml \
#     --frame_range $START_FRAMES $END_FRAMES 1

# echo "=== Running COLMAP to convert model from txt to bin for original data ==="
# for i in $(seq $START_FRAMES $END_FRAMES); do
#     sub_dir=$(printf "%06d" $i)
#     colmap model_converter \
#         --input_path $ORIG_OUT_DIR/$sub_dir/sparse/0 \
#         --output_path $ORIG_OUT_DIR/$sub_dir/sparse/0 \
#         --output_type BIN
# done

echo "=== Running EVC to COLMAP for vsr data ==="
python3 scripts/evc/evc_to_colmap_full.py \
    --data_root $DATA_ROOT \
    --output $VSR_OUT_DIR \
    --image_path exp_data/images_vsr_crop \
    --image_ext .jpg \
    --intri_path intri_vsr_crop.yml \
    --extri_path extri_vsr_crop.yml \
    --frame_range $START_FRAMES $END_FRAMES 1

echo "=== Running COLMAP to convert model from txt to bin for vsr data ==="
for i in $(seq $START_FRAMES $END_FRAMES-1); do
    sub_dir=$(printf "%06d" $i)
    colmap model_converter \
        --input_path $VSR_OUT_DIR/$sub_dir/sparse/0 \
        --output_path $VSR_OUT_DIR/$sub_dir/sparse/0 \
        --output_type BIN
done
