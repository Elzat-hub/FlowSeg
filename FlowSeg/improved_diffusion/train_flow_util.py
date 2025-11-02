import copy
import functools
import os
from pathlib import Path

import blobfile as bf
import numpy as np
import torch as th
import torch.nn.utils as nn_utils
import torch.distributed as dist
from mpi4py import MPI
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn.functional as F

from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .sampling_util import sampling_major_vote_func
from .utils import set_random_seed_for_iterations
from pathlib import Path
import torchvision.utils as tvu
from .boundary_visualization import BoundaryVisualizer

INITIAL_LOG_LOSS_SCALE = 13.0

class FlowTrainLoop:
    def __init__(
        self,
        *,
        model,
        flow_matcher,
        data,
        batch_size, 
        microbatch,  
        lr,
        ema_rate,
        log_interval, 
        save_interval,
        resume_checkpoint,
        logger,
        image_size,
        val_dataset,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        weight_decay=0.0,
        lr_anneal_steps=0,   
        warmup_steps=0,     
        min_lr=0.0,         
        run_without_test=False,
        boundary_loss_weight=0.1,
        args=None,
    ):
        self.visualizer = BoundaryVisualizer(save_root="./training_visualizations")
        self.model = model
        self.flow_matcher = flow_matcher    
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.args = args
        self.logger = logger
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.warmup_steps = warmup_steps  
        self.min_lr = min_lr
        self.boundary_loss_weight = boundary_loss_weight


        self.step = 1
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()   

        self.model_params = list(self.model.parameters())  
        self.master_params = self.model_params    
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE  
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters(self.resume_checkpoint)
        if self.use_fp16:
            self._setup_fp16()
        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.999))

        if self.resume_checkpoint: 
            self._load_optimizer_state(resume_checkpoint)
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate, resume_checkpoint) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128, 
                find_unused_parameters=False,
            )
            self.ema_model = copy.deepcopy(self.model).to(th.device("cpu")) 
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        self.val_dataset = val_dataset
        self.logger = logger
        self.ema_val_best_iou = 0
        self.val_best_iou = 0
        self.val_current_model_name = ""
        self.val_current_model_ema_name = ""
        self.current_model_checkpoint_name = ""
        self.run_without_test = run_without_test

    def _load_and_sync_parameters(self, logs_path): 
        # resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        # model_checkpoint = bf.join(
        #     bf.dirname(logs_path), f"model.pt"
        # )
        logger.log(f"model folder path")
        if logs_path:  
            if Path(logs_path).exists():
                model_path = list(Path(logs_path).glob("model*.pt"))[0]
                self.resume_step = parse_resume_step_from_filename(str(model_path))
                self.step = self.resume_step

                logger.log(f"loading model from checkpoint: {model_path} from step {self.step}...")

                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        str(model_path), map_location=dist_util.dev() 
                    )
                )
        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate, logs_path): 
        ema_params = copy.deepcopy(self.master_params)

        ema_checkpoint = Path(logs_path) / "ema.pt"

        if ema_checkpoint.exists():
            # if dist.get_rank() == 0:
            logger.log(f"loading EMA from checkpoint: {str(ema_checkpoint)}...")
            state_dict = dist_util.load_state_dict(
                str(ema_checkpoint), map_location=dist_util.dev()
            )
            ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)   
        return ema_params  

    def _load_optimizer_state(self, logs_path):  
        opt_checkpoint = Path(logs_path) / "opt.pt"   

        if opt_checkpoint.exists():
            logger.log(f"loading optimizer state from checkpoint: {str(opt_checkpoint)}")
            state_dict = dist_util.load_state_dict(
                str(opt_checkpoint), map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict) 

    def _setup_fp16(self):  
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self, max_iter=250000, start_print_iter=10000, n_rounds=3):
        if dist.get_rank() == 0:
            pbar = tqdm()
        while (self.step < max_iter):
            self.ddp_model.train()
            batch, cond, _ = next(self.data) 
            self.run_step(batch, cond)
            
            if dist.get_rank() == 0:
                pbar.update(1)
            if self.step % self.log_interval == 0 and self.step != 0:
                logger.log(f"interval") 
                logger.log("writing logs...")
                logger.dumpkvs()
                logger.log(f"class {self.args.class_name} lr {self.args.lr}, expansion {self.args.expansion}, "
                           f"rrdb blocks {self.args.rrdb_blocks} gpus {MPI.COMM_WORLD.Get_size()}")
                
            if self.step % self.save_interval == 0:  
                logger.log(f"save model for checkpoint")
                self.save_state_dict()
                dist.barrier()    

            if self.step % self.save_interval == 0 and self.step >= start_print_iter or self.step == 50000:
                if self.run_without_test: 
                    if dist.get_rank() == 0:
                        self.save_checkpoint(self.ema_rate[0], self.ema_params[0], name=f"model")
                else:
                    self.ddp_model.eval()

                    logger.log("starting validation...")
                    output_folder = os.path.join(os.environ["OPENAI_LOGDIR"], f"{self.step}_val_ema")
                    os.makedirs(output_folder, exist_ok=True)

                    self.ema_model = self.ema_model.to(dist_util.dev())
                    self.ema_model.load_state_dict(self._master_params_to_state_dict(self.ema_params[0]))
                    self.ema_model.eval() 

                    ema_val_iou = sampling_major_vote_func(self.flow_matcher, self.ema_model, output_folder=output_folder,
                        dataset=self.val_dataset, logger=self.logger, step=self.step,
                        n_rounds=len(self.val_dataset))
                    self.ema_model = self.ema_model.to(th.device("cpu")) 
                    ema_filename = None
            
                    if dist.get_rank() == 0 :
                        if self.ema_val_best_iou < ema_val_iou:
                            logger.log(f"new best validation IoU: {ema_val_iou} step{self.step}")
                            self.ema_val_best_iou = ema_val_iou
                            ema_filename = self.save_checkpoint(self.ema_rate[0], self.ema_params[0], name=f"val_{ema_val_iou:.7f}")

                            if self.val_current_model_ema_name != "":
                                ckpt_path = bf.join(get_blob_logdir(), self.val_current_model_ema_name)
                                if os.path.exists(ckpt_path):
                                    os.remove(ckpt_path)

                            self.val_current_model_ema_name = ema_filename
     #----------------------添加验证可视化---------------               
                    try:
                        from torch.utils.data import DataLoader
                        val_loader = DataLoader(self.val_dataset, batch_size=1)
                        self.visualizer.comprehensive_visualization(
                        self.ema_model, val_loader, max_samples=3
                        )
                    except Exception as e:
                        logger.log(f"验证可视化过程出错: {e}")
    #-----------------------------------------------------
                set_random_seed_for_iterations(dist.get_rank() + self.step)
                dist.barrier() 
            self.step += 1
            
    def run_step(self, batch, cond): 
        self.forward_backward(batch, cond)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_backward(self, batch, cond):
        zero_grad(self.model_params)  
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i:i + self.microbatch].to(dist_util.dev())
            micro_cond = {  
                k: v[i:i + self.microbatch].to(dist_util.dev()) if isinstance(v, th.Tensor) else v[i:i + self.microbatch]
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0] 

            x1 = micro
            x0 = th.randn_like(micro)
            t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x1)
            model_output = self.ddp_model(t, xt, conditioned_image=micro_cond["conditioned_image"])
            
            if isinstance(model_output, dict):
                vt = model_output['vector_field']
                boundary_preds = model_output.get('boundary_predictions', None)
            else:
                vt = model_output
                boundary_preds = None
            
            flow_loss = th.mean((vt - ut) ** 2)
            print("Flow loss value:", flow_loss.item())

            # 如果有边界预测，计算边界感知损失
            if boundary_preds is not None:
                loss = compute_boundary_aware_loss(boundary_preds, x1, flow_loss, self.boundary_loss_weight)
                print("Total loss with boundary:", loss.item())
            else:
                loss = flow_loss
                print("Flow loss only:", loss.item())

            if last_batch or not self.use_ddp:
                loss = loss.mean()
            else:
                with self.ddp_model.no_sync():
                # loss = loss.mean() / self.microbatch
                    loss = loss.mean()

            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward() 
            else:
                loss.backward()

            logger.logkv_mean("train_loss", loss.item())
            if boundary_preds is not None:
                logger.logkv_mean("flow_loss", flow_loss.item())

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self): 
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
                sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))


    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        current_step = self.step + self.resume_step
    
        if current_step < self.warmup_steps:
            lr = self.min_lr + (self.lr - self.min_lr) * (current_step / self.warmup_steps)
        else:
            adjusted_step = current_step - self.warmup_steps
            adjusted_total = self.lr_anneal_steps - self.warmup_steps

            cosine_decay = 0.5 * (1 + np.cos(np.pi * adjusted_step / adjusted_total))
            lr = self.min_lr + (self.lr - self.min_lr) * cosine_decay
    
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr
    
        logger.logkv("learning_rate", lr)
 
    def log_step(self):
        logger.logkv("step", self.step + self.resume_step) 
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)  
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)
    
    def save_checkpoint(self, rate, params, name):
        state_dict = self._master_params_to_state_dict(params)
        if dist.get_rank() == 0:
            logger.log("saving model {rate}...")
            if not rate:
                filename = f"model_{name}_{(self.step+self.resume_step):06d}.pt"
            else:
                filename = f"ema_{name}_{rate}_{(self.step+self.resume_step):06d}.pt" 
            with bf.BlobFile(bf.join(logger.get_dir(), filename), "wb") as f:
                th.save(state_dict, f)
            return filename

    def save_state_dict(self):

        if dist.get_rank() == 0:
            with bf.BlobFile(bf.join(get_blob_logdir(), f"opt.pt"), "wb",) as f:
                th.save(self.opt.state_dict(), f)

            with bf.BlobFile(bf.join(get_blob_logdir(), f"model{self.step}.pt"), "wb") as f:
                th.save(self._master_params_to_state_dict(self.master_params), f)

            if self.current_model_checkpoint_name != "":
                ckpt_path = bf.join(get_blob_logdir(), self.current_model_checkpoint_name)
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)

            self.current_model_checkpoint_name = bf.join(get_blob_logdir(), f"model{self.step}.pt")

            with bf.BlobFile(bf.join(get_blob_logdir(), f"ema.pt"), "wb") as f:
                th.save(self._master_params_to_state_dict(self.ema_params[0]), f)
        #
        # checkpoint = {
        #     'step': self.step,
        #     'state_dict': self._master_params_to_state_dict(self.master_params),
        #     'ema_state_dict': self._master_params_to_state_dict(self.ema_params[0]),
        #     'optimizer': self.opt.state_dict()
        # }
        #
        # current_model_checkpoint_name = bf.join(get_blob_logdir(), file_name)
        # th.save(checkpoint, current_model_checkpoint_name)
        #
        # if self.current_model_checkpoint_name != "":
        #     ckpt_path = bf.join(get_blob_logdir(), self.current_model_checkpoint_name)
        #     if os.path.exists(ckpt_path):
        #         os.remove(ckpt_path)
        #
        # self.current_model_checkpoint_name = current_model_checkpoint_name

    def save(self, name):
        """保存命名检查点"""
        filename = self.save_checkpoint(0, self.master_params, name)
        for rate, params in zip(self.ema_rate, self.ema_params):
            filename_ema = self.save_checkpoint(rate, params, name)
        
        # if dist.get_rank() == 0:
        #     with bf.BlobFile(
        #         bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
        #         "wb",
        #     ) as f:
        #         th.save(self.opt.state_dict(), f)

        # dist.barrier()

        return filename, filename_ema

    def _master_params_to_state_dict(self, master_params):
        """将master参数转换为状态字典"""
        if self.use_fp16:
            master_params = unflatten_master_params(
                list(self.model.parameters()), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict
    
    def _state_dict_to_master_params(self, state_dict):
        """将状态字典转换为master参数"""
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params
    
def parse_resume_step_from_filename(filename):
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0

def get_blob_logdir():
    return os.environ.get("FLOW_MATCHING_BLOB_LOGDIR", logger.get_dir())

def find_resume_checkpoint():
    return None

def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None

def log_loss_dict(flow_matcher, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles)
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t) 
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

def extract_multi_scale_boundaries(gt_mask, target_scales=[128, 64, 32, 16]):
    """从GT mask提取多尺度边界标签"""
    boundaries = []
    
    # 拉普拉斯核用于边界检测
    laplacian_kernel = th.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
                                   dtype=th.float32, device=gt_mask.device).view(1, 1, 3, 3)
    
    for scale in target_scales:
        # 下采样GT到目标尺度
        if gt_mask.shape[2] != scale:
            mask_scaled = F.interpolate(gt_mask, size=(scale, scale), mode='nearest')
        else:
            mask_scaled = gt_mask
            
        # 拉普拉斯边界检测
        boundary = th.abs(F.conv2d(mask_scaled, laplacian_kernel, padding=1))
        
        # 二值化边界
        boundary = (boundary > 0.1).float()
        
        boundaries.append(boundary)
    
    return boundaries

def compute_boundary_aware_loss(boundary_predictions, gt_mask, flow_loss, boundary_weight=0.1):
    """计算边界感知损失"""
    if boundary_predictions is None or len(boundary_predictions) == 0:
        return flow_loss
    
    # 提取多尺度GT边界
    gt_boundaries = extract_multi_scale_boundaries(gt_mask)
    
    # 多尺度边界损失
    boundary_loss = 0
    for pred, gt_boundary in zip(boundary_predictions, gt_boundaries):
        # 如果预测尺寸与GT不匹配，上采样预测
        if pred.shape[2:] != gt_boundary.shape[2:]:
            pred_upsampled = F.interpolate(pred, size=gt_boundary.shape[2:], mode='bilinear', align_corners=False)
        else:
            pred_upsampled = pred
            
        # 边界检测损失
        boundary_loss += F.binary_cross_entropy_with_logits(pred_upsampled, gt_boundary)
    
    # 平均多尺度损失
    boundary_loss = boundary_loss / len(boundary_predictions)
    
    # 总损失
    total_loss = flow_loss + boundary_weight * boundary_loss
    
    return total_loss