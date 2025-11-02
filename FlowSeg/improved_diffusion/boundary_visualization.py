import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torchvision.utils as tvu
from PIL import Image

class BoundaryVisualizer:
    def __init__(self, save_root="./visualizations"):
        """
        初始化边界可视化器
        Args:
            save_root: 保存可视化结果的根目录
        """
        self.save_root = Path(save_root)
        self.save_root.mkdir(parents=True, exist_ok=True)
        
    def visualize_gt_boundary_extraction(self, gt_mask, save_name, target_scales=[128, 64, 32, 16]):
        """
        可视化GT mask的多尺度边界提取过程
        Args:
            gt_mask: GT mask tensor [B, C, H, W]
            save_name: 保存文件名
            target_scales: 目标尺度列表
        """
        from improved_diffusion.train_flow_util import extract_multi_scale_boundaries
        
        save_dir = self.save_root / "gt_boundaries"
        save_dir.mkdir(exist_ok=True)
        
        with torch.no_grad():
            # 提取多尺度边界
            boundaries = extract_multi_scale_boundaries(gt_mask, target_scales)
            
            # 创建可视化
            fig, axes = plt.subplots(2, len(target_scales) + 1, figsize=(20, 8))
            
            # 第一行：原始GT和各尺度下采样
            axes[0, 0].imshow(gt_mask[0, 0].cpu().numpy(), cmap='gray')
            axes[0, 0].set_title('Original GT Mask')
            axes[0, 0].axis('off')
            
            for i, scale in enumerate(target_scales):
                # 下采样GT到目标尺度
                if gt_mask.shape[2] != scale:
                    mask_scaled = F.interpolate(gt_mask, size=(scale, scale), mode='nearest')
                else:
                    mask_scaled = gt_mask
                
                axes[0, i+1].imshow(mask_scaled[0, 0].cpu().numpy(), cmap='gray')
                axes[0, i+1].set_title(f'GT Mask {scale}x{scale}')
                axes[0, i+1].axis('off')
            
            # 第二行：提取的边界图
            axes[1, 0].axis('off')  # 空白
            for i, (boundary, scale) in enumerate(zip(boundaries, target_scales)):
                boundary_np = boundary[0, 0].cpu().numpy()
                im = axes[1, i+1].imshow(boundary_np, cmap='hot', vmin=0, vmax=1)
                axes[1, i+1].set_title(f'Boundary {scale}x{scale}')
                axes[1, i+1].axis('off')
                
                # 添加colorbar
                plt.colorbar(im, ax=axes[1, i+1], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.savefig(save_dir / f"{save_name}_gt_boundaries.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # 单独保存每个尺度的边界图
            for i, (boundary, scale) in enumerate(zip(boundaries, target_scales)):
                boundary_np = (boundary[0, 0].cpu().numpy() * 255).astype(np.uint8)
                boundary_img = Image.fromarray(boundary_np)
                boundary_img.save(save_dir / f"{save_name}_boundary_{scale}x{scale}.png")
            
            print(f"GT边界提取可视化已保存到: {save_dir}")
    
    def visualize_boundary_detection_features(self, model, input_image, save_name):
        """
        可视化边界检测流的特征图
        Args:
            model: 模型实例
            input_image: 输入图像 [B, C, H, W]
            save_name: 保存文件名
        """
        if not hasattr(model, 'boundary_stream') or model.boundary_stream is None:
            print("模型没有边界检测流，跳过可视化")
            return
            
        save_dir = self.save_root / "boundary_features"
        save_dir.mkdir(exist_ok=True)
        
        model.eval()
        with torch.no_grad():
            # 获取边界检测输出
            boundary_outputs = model.boundary_stream(input_image)
            boundary_features = boundary_outputs['boundary_features']
            boundary_predictions = boundary_outputs['boundary_predictions']
            
            # 可视化特征图
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            
            # 第一行：原始输入图像
            if input_image.shape[1] == 3:  # RGB图像
                img_np = input_image[0].permute(1, 2, 0).cpu().numpy()
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            else:  # 灰度图像
                img_np = input_image[0, 0].cpu().numpy()
            
            axes[0, 0].imshow(img_np, cmap='gray' if input_image.shape[1] == 1 else None)
            axes[0, 0].set_title('Input Image')
            axes[0, 0].axis('off')
            
            # 清空其他位置
            for i in range(1, 4):
                axes[0, i].axis('off')
            
            # 第二行：各阶段特征图（取前几个通道的平均）
            for i, feat in enumerate(boundary_features):
                if i >= 4:
                    break
                    
                # 取前16个通道的平均作为可视化
                feat_vis = feat[0, :16].mean(dim=0).cpu().numpy()
                im = axes[1, i].imshow(feat_vis, cmap='viridis')
                axes[1, i].set_title(f'Stage {i+1} Features\n{feat.shape[2]}x{feat.shape[3]}')
                axes[1, i].axis('off')
                plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
            
            # 第三行：边界预测
            for i, pred in enumerate(boundary_predictions):
                if i >= 4:
                    break
                    
                boundary_map = torch.sigmoid(pred[0, 0]).cpu().numpy()
                im = axes[2, i].imshow(boundary_map, cmap='hot', vmin=0, vmax=1)
                axes[2, i].set_title(f'Boundary Pred\n{pred.shape[2]}x{pred.shape[3]}')
                axes[2, i].axis('off')
                plt.colorbar(im, ax=axes[2, i], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.savefig(save_dir / f"{save_name}_boundary_features.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # 单独保存高分辨率的边界预测图
            for i, pred in enumerate(boundary_predictions):
                boundary_map = torch.sigmoid(pred[0, 0]).cpu().numpy()
                boundary_map = (boundary_map * 255).astype(np.uint8)
                boundary_img = Image.fromarray(boundary_map)
                boundary_img.save(save_dir / f"{save_name}_boundary_pred_stage{i+1}.png")
            
            print(f"边界检测特征可视化已保存到: {save_dir}")
    
    def visualize_gated_fusion_attention(self, model, main_feat, boundary_feat, save_name, stage_name):
        """
        可视化门控融合的注意力权重
        Args:
            model: 模型实例
            main_feat: 主特征
            boundary_feat: 边界特征
            save_name: 保存文件名
            stage_name: 阶段名称
        """
        save_dir = self.save_root / "fusion_attention"
        save_dir.mkdir(exist_ok=True)
        
        # 找到对应的融合模块
        fusion_scale = str(main_feat.shape[2])  # 根据特征图大小确定尺度
        if (hasattr(model, 'boundary_fusions') and 
            model.boundary_fusions is not None and 
            fusion_scale in model.boundary_fusions):
            
            fusion_module = model.boundary_fusions[fusion_scale]
            
            with torch.no_grad():
                # 获取融合过程的中间结果
                boundary_aligned = fusion_module.boundary_align(boundary_feat)
                
                if boundary_aligned.shape[2:] != main_feat.shape[2:]:
                    boundary_aligned = F.interpolate(
                        boundary_aligned, size=main_feat.shape[2:], 
                        mode='bilinear', align_corners=False
                    )
                
                combined = torch.cat([main_feat, boundary_aligned], dim=1)
                gate = fusion_module.gate_network(combined)  # 注意力权重
                fused_feat = fusion_module.fusion_network(combined)
                
                # 可视化
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
                # 第一行：输入特征
                main_vis = main_feat[0, :16].mean(dim=0).cpu().numpy()
                boundary_vis = boundary_aligned[0, :16].mean(dim=0).cpu().numpy()
                
                im1 = axes[0, 0].imshow(main_vis, cmap='viridis')
                axes[0, 0].set_title('Main Features')
                axes[0, 0].axis('off')
                plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
                
                im2 = axes[0, 1].imshow(boundary_vis, cmap='plasma')
                axes[0, 1].set_title('Boundary Features')
                axes[0, 1].axis('off')
                plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
                
                # 注意力权重
                gate_vis = gate[0, :16].mean(dim=0).cpu().numpy()
                im3 = axes[0, 2].imshow(gate_vis, cmap='hot', vmin=0, vmax=1)
                axes[0, 2].set_title('Attention Weights (Gate)')
                axes[0, 2].axis('off')
                plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
                
                # 第二行：融合结果
                fused_vis = fused_feat[0, :16].mean(dim=0).cpu().numpy()
                final_output = main_feat + gate * fused_feat
                final_vis = final_output[0, :16].mean(dim=0).cpu().numpy()
                
                im4 = axes[1, 0].imshow(fused_vis, cmap='viridis')
                axes[1, 0].set_title('Fused Features')
                axes[1, 0].axis('off')
                plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)
                
                im5 = axes[1, 1].imshow(final_vis, cmap='viridis')
                axes[1, 1].set_title('Final Output')
                axes[1, 1].axis('off')
                plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)
                
                # 注意力权重的直方图
                axes[1, 2].hist(gate_vis.flatten(), bins=50, alpha=0.7, color='red')
                axes[1, 2].set_title('Attention Weight Distribution')
                axes[1, 2].set_xlabel('Attention Value')
                axes[1, 2].set_ylabel('Frequency')
                
                plt.tight_layout()
                plt.savefig(save_dir / f"{save_name}_{stage_name}_fusion_attention.png", 
                           dpi=150, bbox_inches='tight')
                plt.close()
                
                # 单独保存注意力热图
                attention_heatmap = (gate_vis * 255).astype(np.uint8)
                attention_img = Image.fromarray(attention_heatmap)
                attention_img.save(save_dir / f"{save_name}_{stage_name}_attention_heatmap.png")
                
                print(f"融合注意力可视化已保存到: {save_dir}")
    
    def comprehensive_visualization(self, model, dataloader, max_samples=5):
        """
        全面的边界可视化分析
        Args:
            model: 模型实例
            dataloader: 数据加载器
            max_samples: 最大可视化样本数
        """
        model.eval()
        
        print(f"开始全面边界可视化分析，最多处理 {max_samples} 个样本...")
        
        with torch.no_grad():
            for i, (gt_mask, cond, name) in enumerate(dataloader):
                if i >= max_samples:
                    break
                
                sample_name = name[0] if isinstance(name, (list, tuple)) else str(i)
                print(f"处理样本: {sample_name}")
                
                # 1. GT边界提取可视化
                self.visualize_gt_boundary_extraction(gt_mask, sample_name)
                
                # 2. 边界检测流特征可视化
                condition_image = cond["conditioned_image"]
                self.visualize_boundary_detection_features(model, condition_image, sample_name)
                
                # 3. 如果可能，可视化模型的完整前向过程
                if hasattr(model, 'boundary_stream') and model.boundary_stream:
                    try:
                        # 模拟一个时间步
                        t = torch.zeros(gt_mask.shape[0], device=gt_mask.device)
                        
                        # 获取边界特征用于融合可视化
                        boundary_outputs = model.boundary_stream(condition_image)
                        boundary_features = boundary_outputs['boundary_features']
                        
                        # 可视化不同尺度的融合注意力（需要修改模型来获取中间特征）
                        print(f"样本 {sample_name} 的边界特征提取完成")
                        
                    except Exception as e:
                        print(f"模型前向过程可视化失败: {e}")
        
        print(f"全面可视化分析完成，结果保存在: {self.save_root}")
        
        # 创建一个总结报告
        self.create_summary_report(max_samples)
    
    def create_summary_report(self, num_samples):
        """创建可视化总结报告"""
        report_path = self.save_root / "visualization_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("边界感知Flow Matching模型可视化报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"处理样本数量: {num_samples}\n")
            f.write(f"保存目录: {self.save_root}\n\n")
            
            f.write("可视化内容:\n")
            f.write("1. GT边界提取 (gt_boundaries/)\n")
            f.write("   - 多尺度边界提取过程\n")
            f.write("   - 各尺度边界图单独保存\n\n")
            
            f.write("2. 边界检测特征 (boundary_features/)\n")
            f.write("   - 各阶段特征图可视化\n")
            f.write("   - 边界预测结果\n")
            f.write("   - 高分辨率边界预测图\n\n")
            
            f.write("3. 融合注意力 (fusion_attention/)\n")
            f.write("   - 门控融合的注意力权重\n")
            f.write("   - 注意力权重分布\n")
            f.write("   - 融合前后特征对比\n\n")
            
            f.write("使用建议:\n")
            f.write("- 检查GT边界提取是否合理\n")
            f.write("- 观察边界检测特征的质量\n")
            f.write("- 分析注意力权重的分布模式\n")
            f.write("- 对比不同样本的特征表现\n")
        
        print(f"可视化报告已保存到: {report_path}")