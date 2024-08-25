import os
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import foolbox
from owndata import CustomDataset,SingleSampleDataset
import torchvision.transforms.functional as tf
import random
import time
import psutil
from model import resnet50
import cv2
import numpy as np

# def qianyi(path):
    # 读取图像并转换为0-1范围的浮点数，同时调整大小为128x128
def load_image(image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))  # 调整大小为128x128
        img = img.astype(np.float32) / 255.0
        return img

    # 图像复制
def imagecopy(path, list):
        image = path.split('/')[-1].split('_fgsm')[0]+'.png'
        original_image_path = f'D:/Study/experiment/CNN/datasets/TSRD/flat_dataset2/{image}'
        adversarial_image_path = f'D:/Study/experiment/CNN/improve/fgsm_end_modified_random/{image.split(".")[0]}_fgsm_0.png'
        for other_image_path in list:
            # 加载图像
            original_image = load_image(original_image_path)
            adversarial_image = load_image(adversarial_image_path)
            other_image = load_image(other_image_path)
            # 提取扰动
            perturbation = adversarial_image - original_image
            # 将扰动添加到其他图片上
            modified_other_image = other_image + perturbation
            # 确保修改后的图像值在有效范围内（0-1）
            modified_other_image = np.clip(modified_other_image, 0, 1)
            # 保存修改后的图片
            modified_other_image_uint8 = (modified_other_image * 255).astype(np.uint8)
            modified_other_image_bgr = cv2.cvtColor(modified_other_image_uint8, cv2.COLOR_RGB2BGR)
            filename = other_image_path.split('/')[-1]
            modified_image_path = f'D:/Study/experiment/CNN/improve/fgsm_end_modified/{filename}'
            cv2.imwrite(modified_image_path, modified_other_image_bgr)
            print(f'{filename}已保存')


def attack_image(device, fmodel, attack, data_loader):
    for (filename,), image, label in data_loader:
        # 随机选择一张图像
        # 生一批成对抗样本(每张图像生成100个对抗样本)
        for i in range(0, 1):
            # 随机epsilons=random.uniform(0.1, 0.15)
            raw, clipped, is_adv = attack(fmodel, image.to(device), label.to(device), epsilons=random.uniform(0.1, 0.15))
            # 将 PyTorch 张量转换为 PIL 图像，保存到本地
            adv_image_pil = tf.to_pil_image(clipped[0])  # 选择第一个批次的图像
            # 保存图像
            save_dir = 'fgsm_end_modified_random/'
            os.makedirs(save_dir, exist_ok=True)
            # 根据标签命名图像文件
            # save_path = f"bim/{filename.split('.')[0]}_bim_{label.item()}_{i}.png"
            save_path = f"fgsm_end_modified_random/{filename.split('.')[0]}_fgsm_{i}.png"
            adv_image_pil.save(save_path)
            print(f"对抗样本已保存至：{save_path}")
            image_path = 'D:\Study\experiment\CNN\datasets\TSRD/flat_dataset2/'
            white = []
            blue = []
            brown = []
            #  将目录下的图片路径保存,以不同颜色背景保存，白色、蓝色、棕色
            mylist = [os.path.join(image_path, i) for i in os.listdir(image_path) if i.endswith("png")]
            for i in mylist:
                if i.split('/')[-1].split('_')[0] in ['000', '001', '002', '003', '004', '005', '006', '007', '008',
                                                      '009',
                                                      '010', '011', '012', '013', '014', '015', '016', '017', '018',
                                                      '019']:
                    white.append(i)
                if i.split('/')[-1].split('_')[0] in ['020', '021', '022', '023', '024', '025', '026', '027', '028',
                                                      '029',
                                                      '030', '031']:
                    blue.append(i)
                if i.split('/')[-1].split('_')[0] in ['032', '033', '034', '035', '036', '037', '038', '039', '040', '041',
                                                      '042',
                                                      '043', '044', '045', '046', '047', '048', '049', '050', '051',
                                                      '052', '053', '054', '055', '056', '057']:
                    brown.append(i)
            if save_path.split('/')[-1].split('_')[0] in ['000', '001', '002', '003', '004', '005', '006', '007', '008',
                                                      '009',
                                                      '010', '011', '012', '013', '014', '015', '016', '017', '018',
                                                      '019']:
                imagecopy(save_path, white)
            if save_path.split('/')[-1].split('_')[0] in ['020', '021', '022', '023', '024', '025', '026', '027', '028',
                                                      '029',
                                                      '030', '031']:
                imagecopy(save_path, blue)
            if save_path.split('/')[-1].split('_')[0] in ['032', '033', '034', '035', '036', '037', '038', '039', '040', '041',
                                                      '042',
                                                      '043', '044', '045', '046', '047', '048', '049', '050', '051',
                                                      '052', '053', '054', '055', '056', '057']:
                imagecopy(save_path, brown)


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
    # dataset_path = 'D:/Study/experiment/CNN/datasets/TSRD/foolbox/'
    white_dataset_path = 'D:\Study\experiment\CNN\datasets\TSRD\white/'
    blue_dataset_path = 'D:\Study\experiment\CNN\datasets\TSRD/blue/'
    brown_dataset_path = 'D:\Study\experiment\CNN\datasets\TSRD/brown/'
    # 提前对图像进行处理，将所有图片转换成相同大小
    transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 将图像大小调整为128x128
    transforms.ToTensor(),  # 将图像转换为PyTorch张量
])
    # custom_dataset = CustomDataset(root_dir=dataset_path, transform=transform)
    white_dataset = CustomDataset(root_dir=white_dataset_path, transform=transform)
    blue_dataset = CustomDataset(root_dir=blue_dataset_path, transform=transform)
    brown_dataset = CustomDataset(root_dir=brown_dataset_path, transform=transform)
    # 创建数据加载器
    batch_size = 1  # 逐一加载图像

    # 从每一类中随机提取一张图片
    white_random_index = random.randint(0, len(white_dataset) - 1)
    blue_random_index = random.randint(0, len(blue_dataset) - 1)
    brown_random_index = random.randint(0, len(brown_dataset) - 1)
    white_single_sample_dataset = SingleSampleDataset(white_dataset[white_random_index])
    blue_single_sample_dataset = SingleSampleDataset(blue_dataset[blue_random_index])
    brown_single_sample_dataset = SingleSampleDataset(brown_dataset[brown_random_index])

    # data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)
    white_data_loader = torch.utils.data.DataLoader(white_single_sample_dataset, batch_size=batch_size, shuffle=False)
    blue_data_loader = torch.utils.data.DataLoader(blue_single_sample_dataset, batch_size=batch_size, shuffle=False)
    brown_data_loader = torch.utils.data.DataLoader(brown_single_sample_dataset, batch_size=batch_size, shuffle=False)

    # 初始化 Foolbox 攻击
    preprocessing = dict(mean=[0.5], std=[0.5], axis=-3)
    fmodel = foolbox.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    attack = foolbox.attacks.LinfFastGradientAttack()

    # 对每张图像进行攻击
    attack_image(device, fmodel, attack, white_data_loader)
    attack_image(device, fmodel, attack, blue_data_loader)
    attack_image(device, fmodel, attack, brown_data_loader)


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
