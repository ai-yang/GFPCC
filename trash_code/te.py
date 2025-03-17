import torch

# 创建一个随机的多维张量
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 扁平化张量
flat_tensor = tensor.view(-1)

# 在扁平化后的张量中找到最大值及其下标
max_value, max_index = torch.max(flat_tensor, 0)

# 获取下标的值
max_index = max_index.item()

print(f"最大值: {max_value}")
print(f"最大值的全局下标: {max_index}")
