import os
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import foolbox
from owndata import CustomDataset
import torchvision.transforms.functional as tf
import random
import time
import psutil
from model import resnet50


def attack_start():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # create model
    model = resnet50(num_classes=58).to(device)
    # load model weights
    weights_path = "D:/Study/experiment/CNN/weights/resNet50_2.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    # 数据集路径
    dataset_path = 'D:\Study\experiment\CNN\datasets\TSRD/new_dataset2/'
    # 提前对图像进行处理，将所有图片转换成相同大小
    transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 将图像大小调整为128x128
    transforms.ToTensor(),  # 将图像转换为PyTorch张量
])
    custom_dataset = CustomDataset(root_dir=dataset_path, transform=transform)
    # 创建数据加载器
    batch_size = 1  # 逐一加载图像
    data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)
    # 初始化 Foolbox 攻击
    preprocessing = dict(mean=[0.5], std=[0.5], axis=-3)
    fmodel = foolbox.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    attack = foolbox.attacks.LinfFastGradientAttack()

    # 对每张图像进行攻击
    for (filename,), image, label in data_loader:
        # 生一批成对抗样本(每张图像生成100个对抗样本)
        for i in range(0, 1):
            # 随机epsilons=random.uniform(0.1, 0.15)
            raw, clipped, is_adv = attack(fmodel, image.to(device), label.to(device), epsilons=random.uniform(0.1, 0.15))
            # 将 PyTorch 张量转换为 PIL 图像，保存到本地
            adv_image_pil = tf.to_pil_image(clipped[0])  # 选择第一个批次的图像
            # 保存图像
            save_dir = 'fgsm_end/'
            os.makedirs(save_dir, exist_ok=True)
            # 根据标签命名图像文件
            # save_path = f"pgd/{filename.split('.')[0]}_pgd_{label.item()}_{i}.png"
            save_path = f"fgsm_end/{filename.split('.')[0]}_fgsm_{i}.png"
            adv_image_pil.save(save_path)
            print(f"对抗样本已保存至：{save_path}")


def main():
    # 开始时间
    start_time = time.time()
    # 记录GPU初始内存
    initial_memory_allocated = torch.cuda.memory_allocated()
    initial_memory_cached = torch.cuda.memory_reserved()
    # 记录初始内存使用情况
    initial_system_memory = psutil.virtual_memory().used / (1024 ** 2)

    # 开始攻击
    attack_start()

    # 记录结束时间
    end_time = time.time()
    # 记录结束时GPU内存
    final_memory_allocated = torch.cuda.memory_allocated()
    final_memory_cached = torch.cuda.memory_reserved()
    # 记录结束内存使用情况
    final_system_memory = psutil.virtual_memory().used / (1024 ** 2)
    # 打印耗时
    print(f"Execution time: {end_time - start_time} seconds")
    # 打印GPU内存消耗情况
    print(f"Initial GPU memory allocated: {initial_memory_allocated / 1024 ** 2:.2f} MB")
    print(f"Initial GPU memory cached: {initial_memory_cached / 1024 ** 2:.2f} MB")
    print(f"Final GPU memory allocated: {final_memory_allocated / 1024 ** 2:.2f} MB")
    print(f"Final GPU memory cached: {final_memory_cached / 1024 ** 2:.2f} MB")
    # 打印系统内存消耗情况
    print(f"Initial system memory used: {initial_system_memory:.2f} MB")
    print(f"Final system memory used: {final_system_memory:.2f} MB")
    print(f"All system memory used: {final_system_memory - initial_system_memory:.2f} MB")


if __name__ == '__main__':
    main()
