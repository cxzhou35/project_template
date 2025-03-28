N_VIEWS=23
START_FRAMES=200
END_FRAMES=229
SR_RES=128p
DATA_ROOT='/home/zhouchenxu/datasets/vsr_4dv/shijie_far'
ORIG_OUT_DIR='/home/zhouchenxu/codes/gsplat/data/shijie_face_close211/orig'
VSR_OUT_DIR="/home/zhouchenxu/codes/gsplat/data/shijie_far/vsr_from_cropped"

# python3 generate_data_lod.py --root_dir $DATA_ROOT --gen_lr --lr_res 1080p
# python3 generate_data_lod.py --root_dir $DATA_ROOT --gen_lr --lr_res $SR_RES

# run VSR model get SR video then run the following command
# python3 generate_data_lod.py --root_dir $DATA_ROOT --gen_sr --lr_res $SR_RES

# # run evc convert script
# echo "=== Running EVC to COLMAP for original data ==="
# python3 evc_to_colmap_full.py -d $DATA_ROOT -o $ORIG_OUT_DIR --camera_model PINHOLE -t orig --frame_range 0 30 1

# echo "=== Running EVC to COLMAP for vsr data ==="
python3 evc_to_colmap_full.py -d $DATA_ROOT -o $VSR_OUT_DIR --camera_model PINHOLE -t vsr --frame_range 200 260 1 --lr_res $SR_RES

# run colmap to convert model from txt to bin
echo "=== Running COLMAP to convert model from txt to bin for original data ==="
for i in $(seq $START_FRAMES $END_FRAMES); do
    sub_dir=$(printf "%06d" $i)
    # colmap model_converter --input_path $ORIG_OUT_DIR/$sub_dir/sparse/0 --output_path $ORIG_OUT_DIR/$sub_dir/sparse/0 --output_type BIN
    colmap model_converter --input_path $VSR_OUT_DIR/$sub_dir/sparse/0 --output_path $VSR_OUT_DIR/$sub_dir/sparse/0 --output_type BIN
done
