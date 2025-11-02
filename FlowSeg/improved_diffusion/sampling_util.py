import math
import os
import time

from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.utils as tvu
from PIL import Image
from kornia import denormalize
from torchdyn.core import NeuralODE
from sklearn.metrics import f1_score, jaccard_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from . import dist_util
from .metrics import FBound_metric, WCov_metric, HD_metric 
from Datasets.monu import MonuDataset
from Datasets.dataset_prostate import Prostate2DDataset
from Datasets.isic import ISICDataset
from .utils import set_random_seed_for_iterations
import nibabel as nib  
from Datasets.prostate_data_process import save_2d_slice_nii
from thop import profile
 
cityspallete = [
    0, 0, 0,
    128, 64, 128,
    244, 35, 232,
    70, 70, 70,
    102, 102, 156,
    190, 153, 153,
    153, 153, 153,
    250, 170, 30,
    220, 220, 0,
    107, 142, 35,
    152, 251, 152,
    0, 130, 180,
    220, 20, 60,
    255, 0, 0,
    0, 0, 142,
    0, 0, 70,
    0, 60, 100,
    0, 80, 100,
    0, 0, 230,
    119, 11, 32,
]


def calculate_metrics(x, gt):
    """计算各种评估指标"""
    predict = x.detach().cpu().numpy().astype('uint8')
    target = gt.detach().cpu().numpy().astype('uint8')
    return f1_score(target.flatten(), predict.flatten()), \
           jaccard_score(target.flatten(), predict.flatten()), \
           WCov_metric(predict, target), \
           FBound_metric(predict, target), \
            HD_metric(predict, target)

def flow_based_sampling(flow_matcher, model, x0, condition, steps=2):
    device = condition.device
    
    def ode_fn(t, x, args=None):
        
        t_scalar = t.item() if torch.is_tensor(t) else t
        t_batch = torch.full((x.shape[0],), t_scalar, device=device, dtype=x.dtype)
        with torch.no_grad():
            model_output = model(timesteps=t_batch, x=x, conditioned_image=condition)
            # Check if the output is a dictionary and extract the vector field
            if isinstance(model_output, dict):
                return model_output['vector_field']
            else:
                return model_output
    
    node = NeuralODE(ode_fn, solver="euler")
    with torch.no_grad():
        traj = node.trajectory(
            x0,
            t_span=torch.linspace(0, 1, steps, device=device),
        )
    print(traj.shape)
    print(traj[-1].shape)
    return traj[-1]

def sampling_major_vote_func(flow_matcher, ddp_model, output_folder, dataset, logger, step, n_rounds=3):
    """主要的采样和评估函数"""
    ddp_model.eval()
    batch_size = 1
    major_vote_number = 10
    loader = DataLoader(dataset, batch_size=batch_size)
    loader_iter = iter(loader)

    f1_score_list = []
    miou_list = []
    fbound_list = []
    wcov_list = []
    hd_list = []

    os.makedirs(output_folder, exist_ok=True)

    # -------------------------------------------------------------
    # 在这里计算 FLOPs 和参数量
    # 确保在开始测量前清空所有 GPU 内存统计
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(dist_util.dev())
        torch.cuda.synchronize()

    # logger.info("开始计算模型 FLOPs 和参数量...")
    # if hasattr(ddp_model, 'module'):
    #     model_to_profile = ddp_model.module
    # else:
    #     model_to_profile = ddp_model
    # # 获取一个样本来模拟推理过程
    # with torch.no_grad():
    #     gt_mask, condition_on, name = next(loader_iter)
    
    # condition_on = condition_on["conditioned_image"].to(dist_util.dev())
    # # 模拟输入 x，大小需与实际推理一致
    # x_dummy = torch.randn(
    #     batch_size, 
    #     model_to_profile.in_channels, 
    #     condition_on.shape[2], 
    #     condition_on.shape[3], 
    #     device=condition_on.device
    # )

    # timesteps_dummy = torch.full((batch_size,), 0.5, device=x_dummy.device)
    # # **修正：不再向 inputs 中添加 y_dummy**
    # # **因为你的模型配置是非类别条件（class_cond=False），所以 y 应该为 None**
    # flops, params = profile(
    #     model_to_profile,
    #     # 确保输入元组的顺序和 forward 函数的签名 (timesteps, x, y=None, conditioned_image=None) 保持一致
    #     # 由于y是None，可以省略不写
    #     # inputs=(torch.full((batch_size,), 0.5, device=x_dummy.device), x_dummy, condition_on),
    #     inputs=(timesteps_dummy, x_dummy, None, condition_on),
    #     verbose=False
    # )
    # logger.info(f"模型 FLOPs: {flops / 1e9:.2f} G")
    # logger.info(f"模型参数量: {params / 1e6:.2f} M")

    start_time = time.time()
#-------------------------------------------------------------------

    with torch.no_grad():
        for round_index in tqdm(
            range(n_rounds), desc="Generating samples for evaluation"
            ):
            gt_mask, condition_on, name = next(loader_iter)
            set_random_seed_for_iterations(step + int(name[0].split("_")[1]))
            
            gt_mask = (gt_mask + 1.0) / 2.0
            condition_on = condition_on["conditioned_image"]
            former_frame_for_feature_extraction = condition_on.to(dist_util.dev())

            for i in range(gt_mask.shape[0]):
                gt_slice_np = gt_mask[i][0].detach().cpu().numpy().astype(np.uint8) 
                if isinstance(dataset, MonuDataset):
                    gt_img = Image.fromarray((gt_mask[i][0].detach().cpu().numpy() * 255).astype(np.uint8))
                    gt_img.putpalette(cityspallete)
                    gt_img.save(
                        os.path.join(output_folder, f"{name[i]}_gt_palette.png"))
                    # gt_img = Image.fromarray((gt_mask[i][0].detach().cpu().numpy() - 1).astype(np.uint8))
                    gt_img = Image.fromarray((gt_mask[i][0].detach().cpu().numpy() * 255).astype(np.uint8))
                    gt_img.save(
                        os.path.join(output_folder, f"{name[i]}_gt.png"))
                elif isinstance(dataset, ISICDataset):
                    gt_img = Image.fromarray((gt_mask[i][0].detach().cpu().numpy() * 255).astype(np.uint8)) # Create paletted image
                    gt_img.putpalette(cityspallete)
                    gt_img.save(
                        os.path.join(output_folder, f"{name[i]}_gt_palette.png"))
                    # Save plain 0/1 mask PNG
                    gt_img = Image.fromarray((gt_mask[i][0].detach().cpu().numpy() * 255).astype(np.uint8)) 
                    gt_img.save(os.path.join(output_folder, f"{name[i]}_gt.png"))

                else:
                    gt_filename_nii = f"{name[i]}_gt.nii.gz" 
                    save_2d_slice_nii(gt_slice_np, os.path.join(output_folder, gt_filename_nii))

            for i in range(condition_on.shape[0]):
                if isinstance(dataset, MonuDataset):
                    denorm_condition_on = denormalize(condition_on.clone(), mean=dataset.mean, std=dataset.std)
                    tvu.save_image(
                        denorm_condition_on[i,] / 255.,
                        os.path.join(output_folder, f"{name[i]}_condition_on.png")
                    )
                elif isinstance(dataset, ISICDataset):
                    denorm_condition_on = denormalize(condition_on.clone(), mean=dataset.mean, std=dataset.std)
                    tvu.save_image(
                        denorm_condition_on[i,] / 255., # Scale to 0-1 range for tvu.save_image
                        os.path.join(output_folder, f"{name[i]}_condition_on.png")
                    )
                else:
                    condition_slice_np = former_frame_for_feature_extraction[i][0].detach().cpu().numpy() 
                    condition_slice_np_uint8 = (condition_slice_np * 255).astype(np.uint8)
                    condition_filename_nii = f"{name[i]}_condition_on.nii.gz" 
                    save_2d_slice_nii(condition_slice_np_uint8, os.path.join(output_folder, condition_filename_nii))

            if isinstance(dataset, MonuDataset):
                
                _, _, W, H = former_frame_for_feature_extraction.shape
                kernel_size = dataset.image_size
                stride = 256
                patches = []
                for y, x in np.ndindex((((W - kernel_size) // stride) + 1, ((H - kernel_size) // stride) + 1)):
                    y = y * stride
                    x = x * stride
                    patches.append(former_frame_for_feature_extraction[0,
                        :,
                        y: min(y + kernel_size, W),
                        x: min(x + kernel_size, H)])
                patches = torch.stack(patches)

                major_vote_list = []
                for i in range(major_vote_number):
                    x_list = []
                    for index in range(math.ceil(patches.shape[0] / 4)):

                        condition_batch = patches[index * 4: min((index + 1) * 4, patches.shape[0])]
                        model_kwargs = {"conditioned_image": patches[index * 4: min((index + 1) * 4, patches.shape[0])]}

                        x0 = torch.randn(model_kwargs["conditioned_image"].shape[0],
                                         gt_mask.shape[1], 
                                         model_kwargs["conditioned_image"].shape[2], 
                                         model_kwargs["conditioned_image"].shape[3],
                                         device=condition_batch.device)
                        
                        x = flow_based_sampling(flow_matcher, ddp_model, x0, condition_batch)
                        x_list.append(x)
                    out = torch.cat(x_list)

                    
                    output = torch.zeros(1, gt_mask.shape[1], W, H)
                    idx_sum = torch.zeros(1, gt_mask.shape[1], W, H)
                    for index, val in enumerate(out):
                        y, x = np.unravel_index(index, (((W - kernel_size) // stride) + 1, ((H - kernel_size) // stride) + 1))
                        y = y * stride
                        x = x * stride

                        idx_sum[0,
                        :,
                        y: min(y + kernel_size, W),
                        x: min(x + kernel_size, H)] += 1

                        output[0,
                        :,
                        y: min(y + kernel_size, W),
                        x: min(x + kernel_size, H)] += val[:, :min(y + kernel_size, W) - y, :min(x + kernel_size, H) - x].cpu().data.numpy()

                    output = output / idx_sum
                    major_vote_list.append(output)

                x = torch.cat(major_vote_list)
                
            else:
                
                x_list = []
                for _ in range(major_vote_number):
                    x0 = torch.randn(1, 1, 
                                   former_frame_for_feature_extraction.shape[2],
                                   former_frame_for_feature_extraction.shape[3],
                                   device=former_frame_for_feature_extraction.device)
                    x = flow_based_sampling(flow_matcher, ddp_model, x0, former_frame_for_feature_extraction)
                    x_list.append(x)
                x = torch.cat(x_list)

            
            x = (x + 1.0) / 2.0
            if x.shape[2] != gt_mask.shape[2] or x.shape[3] != gt_mask.shape[3]:
                x = F.interpolate(x, gt_mask.shape[2:], mode='bilinear')
            x = torch.clamp(x, 0.0, 1.0)

            
            x = x.mean(dim=0, keepdim=True).round()

            
            for i in range(x.shape[0]):
                out_slice_np = x[i][0].detach().cpu().numpy().astype(np.uint8)
                if isinstance(dataset, MonuDataset):
                    # out_img = Image.fromarray((x[i][0].detach().cpu().numpy() * 255).astype(np.uint8))
                    out_img = Image.fromarray((x[i][0].detach().cpu().numpy()).astype('uint8'))
                    out_img.putpalette(cityspallete)
                    out_img.save(
                        os.path.join(output_folder, f"{name[i]}_model_output_palette.png"))
                    out_img = Image.fromarray((x[i][0].detach().cpu().numpy() * 255).astype(np.uint8))
                    out_img.save(
                        os.path.join(output_folder, f"{name[i]}_model_output.png"))
                elif isinstance(dataset, ISICDataset):
                    out_img = Image.fromarray((x[i][0].detach().cpu().numpy()).astype('uint8'))
                    out_img.putpalette(cityspallete)
                    out_img.save(
                        os.path.join(output_folder, f"{name[i]}_model_output_palette.png"))
                    out_img = Image.fromarray((x[i][0].detach().cpu().numpy() * 255).astype(np.uint8))
                    out_img.save(
                        os.path.join(output_folder, f"{name[i]}_model_output.png"))
                 
                else:
                    
                    out_filename_nii = f"{name[i]}_model_output.nii.gz" 
                    save_2d_slice_nii(out_slice_np, os.path.join(output_folder, out_filename_nii))

            
            for index, (gt_im, out_im) in enumerate(zip(gt_mask, x)):
                f1, miou, wcov, fbound, hd= calculate_metrics(out_im[0], gt_im[0])
                f1_score_list.append(f1)
                miou_list.append(miou)
                wcov_list.append(wcov)
                fbound_list.append(fbound)
                hd_list.append(hd)

                logger.info(
                    f"{name[index]} iou {miou_list[-1]}, f1_Score {f1_score_list[-1]}, "
                    f"WCov {wcov_list[-1]}, boundF {fbound_list[-1]}, HD {hd_list[-1]:.4f}")

    end_time = time.time()
    inference_time = end_time - start_time
    total_samples = n_rounds * major_vote_number
    average_inference_time = inference_time / total_samples

    logger.info(f"对 {total_samples} 个样本的总推理时间: {inference_time:.4f} 秒")
    logger.info(f"平均每个样本的推理时间: {average_inference_time:.4f} 秒")

    if torch.cuda.is_available():
        max_gpu_memory = torch.cuda.max_memory_allocated(dist_util.dev())
        logger.info(f"峰值GPU内存占用: {max_gpu_memory / 1024**2:.2f} MB")
            
    # 确保在函数返回之前执行同步
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    my_length = len(miou_list)
    length_of_data = torch.tensor(len(miou_list), device=dist_util.dev())
    gathered_length_of_data = [torch.tensor(1, device=dist_util.dev()) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_length_of_data, length_of_data)
    max_len = torch.max(torch.stack(gathered_length_of_data))

    
    iou_tensor = torch.tensor(miou_list + [torch.tensor(-1)] * (max_len - my_length), device=dist_util.dev())
    f1_tensor = torch.tensor(f1_score_list + [torch.tensor(-1)] * (max_len - my_length), device=dist_util.dev())
    wcov_tensor = torch.tensor(wcov_list + [torch.tensor(-1)] * (max_len - my_length), device=dist_util.dev())
    boundf_tensor = torch.tensor(fbound_list + [torch.tensor(-1)] * (max_len - my_length), device=dist_util.dev())
    hd_tensor = torch.tensor(hd_list + [torch.tensor(-1)] * (max_len - my_length), device=dist_util.dev())

    
    gathered_miou = [torch.ones_like(iou_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_f1 = [torch.ones_like(f1_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_wcov = [torch.ones_like(wcov_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_boundf = [torch.ones_like(boundf_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_hd = [torch.ones_like(hd_tensor) * -1 for _ in range(dist.get_world_size())]

    
    dist.all_gather(gathered_miou, iou_tensor)
    dist.all_gather(gathered_f1, f1_tensor)
    dist.all_gather(gathered_wcov, wcov_tensor)
    dist.all_gather(gathered_boundf, boundf_tensor)
    dist.all_gather(gathered_hd, hd_tensor)

     
    logger.info("measure total avg")
    gathered_miou = torch.cat(gathered_miou)
    gathered_miou = gathered_miou[gathered_miou != -1]
    logger.info(f"mean iou {gathered_miou.mean()}")

    gathered_f1 = torch.cat(gathered_f1)
    gathered_f1 = gathered_f1[gathered_f1 != -1]
    logger.info(f"mean f1 {gathered_f1.mean()}")

    gathered_wcov = torch.cat(gathered_wcov)
    gathered_wcov = gathered_wcov[gathered_wcov != -1]
    logger.info(f"mean WCov {gathered_wcov.mean()}")

    gathered_boundf = torch.cat(gathered_boundf)
    gathered_boundf = gathered_boundf[gathered_boundf != -1]
    logger.info(f"mean boundF {gathered_boundf.mean()}")
    gathered_hd = torch.cat(gathered_hd)
    gathered_hd = gathered_hd[gathered_hd != -1]
    logger.info(f"mean HD {gathered_hd.mean()}")

    dist.barrier()
    return gathered_miou.mean().item()