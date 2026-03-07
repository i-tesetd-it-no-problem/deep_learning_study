import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# 1. 定义函数：完成图像加载、卷积计算、特征图提取和可视化
def demo01():
    # 2. 读取图像
    # plt.imread() 读取出来通常是 HWC 格式
    # H: 高度(height)
    # W: 宽度(width)
    # C: 通道(channel)
    img = plt.imread("./zhaomeiyan.jpg")

    # 3. 打印原始图像信息
    print("原始图像信息:")
    print(f"shape: {img.shape}, dtype: {img.dtype}")  # 例如: (1440, 1080, 3)

    # 4. 处理图像通道数
    # 有些图片可能是 RGBA(4通道)，这里只保留前3个通道 RGB
    # 如果本身就是 RGB(3通道)，则不会有影响
    if len(img.shape) == 3 and img.shape[2] > 3:
        img = img[:, :, :3]

    # 5. 如果图像数据类型不是 float32，则转成 float32
    # 卷积层通常使用 float32 进行计算
    img_tensor = torch.tensor(img, dtype=torch.float32)

    # 6. 将图像维度从 HWC 转换为 CHW
    # 因为 PyTorch 的卷积层 nn.Conv2d 要求输入格式是 NCHW
    # N: batch size，表示一批图像有多少张
    # C: 通道数
    # H: 高度
    # W: 宽度
    img_tensor = img_tensor.permute(2, 0, 1)

    # 7. 在最前面增加一个 batch 维度
    # 从 (C, H, W) 变成 (N, C, H, W)
    # 这里 N=1，表示只有一张图片
    img_tensor = img_tensor.unsqueeze(0)

    # 8. 打印转换后的张量信息
    print("\n转换后的张量信息:")
    print(
        f"shape: {img_tensor.shape}, dtype: {img_tensor.dtype}"
    )  # 例如: (1, 3, 1440, 1080)

    # 9. 创建卷积层
    # in_channels=3：输入是 RGB 图像，所以通道数为 3
    # out_channels=4：使用 4 个卷积核，因此输出 4 个特征图
    # kernel_size=3：卷积核大小为 3x3
    # stride=1：步长为 1
    # padding=1：边缘补 1 圈 0，这样输出的高宽和输入保持一致
    conv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1)

    # 10. 进行卷积计算
    # 这里只是做前向传播演示，不需要反向传播，所以使用 no_grad()
    with torch.no_grad():
        conv_img = conv(img_tensor)

    # 11. 打印卷积输出信息
    print("\n卷积输出信息:")
    print(f"shape: {conv_img.shape}, dtype: {conv_img.dtype}")  # 例如: (1, 4, H, W)

    # 12. 可视化 4 个输出特征图
    # conv_img 的形状是 (1, 4, H, W)
    # 第 0 维是 batch，只有 1 张图，所以取 conv_img[0]
    # 得到的形状是 (4, H, W)
    feature_maps = conv_img[0]

    # 13. 创建画布，准备显示 4 个特征图
    plt.figure(figsize=(12, 8))

    # 14. 遍历 4 个通道，每个通道对应一个特征图
    for i in range(feature_maps.shape[0]):
        # 取出第 i 个特征图
        # feature_map 的形状是 (H, W)
        feature_map = feature_maps[i].detach().cpu()

        # 在 2x2 的子图中显示第 i+1 个特征图
        plt.subplot(2, 2, i + 1)

        # 显示特征图
        # cmap="gray" 表示使用灰度图显示，更适合观察特征响应强弱
        # plt.imshow(feature_map, cmap="gray")
        plt.imshow(feature_map)

        # 设置标题
        plt.title(f"Feature Map {i + 1}")

        # 关闭坐标轴，画面更简洁
        plt.axis("off")

    # 15. 自动调整子图间距，避免重叠
    plt.tight_layout()

    # 16. 显示图像
    plt.show()


# 17. 程序入口
if __name__ == "__main__":
    demo01()
