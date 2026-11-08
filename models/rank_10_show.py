import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_rank_10_images(query_image_path, ranked_image_paths):
    # 读取查询图片

    query_image = cv2.imread(query_image_path)
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
    query_image = cv2.resize(query_image, (144,288))
    # 读取排名前 10 的图片
    ranked_image_paths= ranked_image_paths[:10]
    ranked_images = []
    for path in ranked_image_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (144, 288))

        ranked_images.append(img)

    # 创建一个子图布局
    fig, axes = plt.subplots(1, 11, figsize=(20, 5))

    # 显示查询图片
    axes[0].imshow(query_image)
    axes[0].set_title('Query Image')
    axes[0].axis('off')

    # 显示排名前 10 的图片
    for i in range(10):
        axes[i + 1].imshow(ranked_images[i])
        axes[i + 1].set_title(f'Rank {i + 1}')
        axes[i + 1].axis('off')

    plt.show()

if __name__ == '__main__':
    # 示例用法
    query_image_path = '../SYSU-MM01/cam3/0001/0001.jpg'
    ranked_image_paths = [
        '../SYSU-MM01/cam1/0001/0001.jpg',
        '../SYSU-MM01/cam1/0001/0002.jpg',
        '../SYSU-MM01/cam1/0001/0003.jpg',
        '../SYSU-MM01/cam1/0001/0004.jpg',
        '../SYSU-MM01/cam1/0001/0005.jpg',
        '../SYSU-MM01/cam3/0001/0006.jpg',
        '../SYSU-MM01/cam1/0001/0007.jpg',
        '../SYSU-MM01/cam1/0001/0008.jpg',
        '../SYSU-MM01/cam1/0001/0009.jpg',
        '../SYSU-MM01/cam1/0001/0010.jpg'
    ]
    first_five_paths = ranked_image_paths[:10]
    print(first_five_paths)
    show_rank_10_images(query_image_path, ranked_image_paths)
