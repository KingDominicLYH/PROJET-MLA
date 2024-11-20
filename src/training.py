import torch
from Encoder import Encoder
from Decoder import Decoder
from Discriminator import Discriminator

# 初始化模型
encoder = Encoder()
decoder = Decoder(n_attributes=5)
discriminator = Discriminator(n_attributes=5)

# 优化器
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=0.0002, betas=(0.5, 0.999)
)

# λE 动态调度
lambda_e = 0.0
lambda_e_max = 0.0001
num_iterations = 500000


def update_lambda_e(current_iteration, max_iterations):
    return min(lambda_e_max, (lambda_e_max / max_iterations) * current_iteration)


# 训练循环
for iteration in range(1, num_iterations + 1):
    lambda_e = update_lambda_e(iteration, num_iterations)
    # 模型前向计算
    latent_representation = encoder(input_image)
    reconstructed_image = decoder(latent_representation, attribute_labels)
    discriminator_output = discriminator(latent_representation)

    # 损失计算
    reconstruction_loss = torch.nn.functional.mse_loss(reconstructed_image, input_image)
    discriminator_loss = -torch.mean(torch.log(discriminator_output + 1e-8))
    total_loss = reconstruction_loss + lambda_e * discriminator_loss

    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # 打印训练状态
    if iteration % 10000 == 0:
        print(f"Iteration {iteration}: λE = {lambda_e:.6f}, Loss = {total_loss.item():.6f}")
