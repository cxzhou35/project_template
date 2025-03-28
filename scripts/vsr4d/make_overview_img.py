import argparse
import math
import os

from PIL import Image, ImageDraw


def create_overview_grid(
    input_folder, output_path, cols=None, thumb_size=(256, 256), margin=0
):
    """
    创建多视角图片的网格概览图

    参数：
    - input_folder: 包含输入图片的文件夹路径
    - output_path: 输出图片的保存路径
    - cols: 网格列数（自动计算行数）
    - thumb_size: 缩略图尺寸 (width, height)
    - margin: 图片间距（像素）
    """

    # 获取所有图片文件
    valid_ext = [".jpg", ".jpeg", ".png", ".bmp"]
    images = [
        f
        for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in valid_ext
    ]

    if not images:
        raise ValueError("未找到支持的图片文件")

    # 自动计算布局
    num_images = len(images)
    if cols is None:
        cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)

    # 创建画布
    thumb_w, thumb_h = thumb_size
    canvas_w = cols * (thumb_w + margin) - margin
    canvas_h = rows * (thumb_h + margin) - margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    # 处理并排列图片
    for i, img_file in enumerate(sorted(images)):  # 按文件名排序
        img_path = os.path.join(input_folder, img_file)

        try:
            img = Image.open(img_path)
            img.thumbnail(thumb_size)  # 保持比例缩放

            # 计算位置
            x = (i % cols) * (thumb_w + margin)
            y = (i // cols) * (thumb_h + margin)

            # 居中粘贴
            paste_w, paste_h = img.size
            offset = ((thumb_w - paste_w) // 2, (thumb_h - paste_h) // 2)
            canvas.paste(img, (x + offset[0], y + offset[1]))

            # 添加编号标签（可选）
            # 在图中标注不要在图上方
            draw.text(
                (x - 5, y + 5),
                f"{i:02d}",
                fill=(255, 0, 0),
                stroke_width=0.5,
            )

        except Exception as e:
            print(f"处理图片 {img_file} 时出错: {str(e)}")
            continue

    canvas.save(output_path)
    print(f"概览图已保存至 {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="输入文件夹路径")
    parser.add_argument("-o", "--output", default="overview.jpg", help="输出路径")
    parser.add_argument("-c", "--cols", type=int, help="指定列数")
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=("WIDTH", "HEIGHT"),
        help="缩略图尺寸",
    )
    args = parser.parse_args()

    create_overview_grid(
        input_folder=args.input,
        output_path=args.output,
        cols=args.cols,
        thumb_size=tuple(args.size),
    )
