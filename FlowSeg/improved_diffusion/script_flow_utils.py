"""
用于创建和配置flow matching模型，并处理模型训练所需的各种配置项
"""
import argparse

from .unet import UNetModel
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher
)

NUM_CLASSES = None

def model_and_flow_matching_defaults():
    """
    Flow matching训练的默认参数。
    """
    return dict(
        dataset_name="isic",    # monu/prostate/isic
        image_size=256,
        num_channels=128,         
        num_res_blocks=2,         
        num_heads=4,              
        num_heads_upsample=-1,    
        attention_resolutions="16,8",  
        dropout=0.0,              
        rrdb_blocks=12,           
        deeper_net=False,         
        learn_sigma=False,        
        sigma_small=False,        
        class_cond=False,         
        class_name="train",       
        expansion=False,          
        use_checkpoint=False,     
        use_scale_shift_norm=True,  
        rrdb_out_channels=None, 
        flow_matching_type="exact_ot",  
        sigma=0.0,               
        seed=None,
        enable_boundary_fusion=True,
        boundary_fusion_scales="128,64,32,16",  # 字符串格式，后续解析
        boundary_loss_weight=0.1,
    )

def create_model_and_flow_matcher(
    dataset_name,
    image_size,
    class_cond,
    learn_sigma,
    sigma_small,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    rrdb_blocks,
    deeper_net,
    class_name,
    expansion,
    use_checkpoint,
    use_scale_shift_norm,
    rrdb_out_channels,
    flow_matching_type,
    sigma,
    seed,
    enable_boundary_fusion,
    boundary_fusion_scales,
    boundary_loss_weight,
):
    """
    创建UNet模型和Flow Matcher.
    """
    _ = seed   
    _ = expansion
    _ = class_name

     
    if image_size == 256:
        if deeper_net:
            channel_mult = (1, 1, 1, 2, 2, 4, 4)
        else:
            channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
     
    for res_str in attention_resolutions.split(","):
        if res_str:  
             
            attention_ds.append(image_size // int(res_str))
    attention_resolutions_ds_tuple = tuple(attention_ds)  

    if boundary_fusion_scales:
        boundary_scales = [int(s) for s in boundary_fusion_scales.split(",") if s]
    else:
        boundary_scales = []
    
    model = UNetModel(
        image_size=image_size,
        in_channels=1,            
        model_channels=num_channels,
        out_channels=1,           
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions_ds_tuple,  
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        rrdb_blocks=rrdb_blocks,
        rrdb_out_channels=rrdb_out_channels,
        enable_boundary_fusion=enable_boundary_fusion,
        boundary_fusion_scales=boundary_scales
    )

     
    if flow_matching_type == "exact_ot":
        flow_matcher = ExactOptimalTransportConditionalFlowMatcher(
            sigma=sigma,
            # reg=ot_reg
        )
    elif flow_matching_type == "target":
        flow_matcher = TargetConditionalFlowMatcher(
            sigma=sigma
        )
    elif flow_matching_type == "basic":
        flow_matcher = ConditionalFlowMatcher(
            sigma=sigma
        )
    else:
        raise ValueError(f"Unknown flow matching type: {flow_matching_type}")

    return model, flow_matcher

def add_dict_to_argparser(parser, default_dict):
    """
    将字典添加到参数解析器。
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        # Special handling for attention_resolutions and cross_attention_resolutions
        # which are passed as strings but need to be parsed later
        if k == "attention_resolutions" or k == "cross_attention_resolutions":
            parser.add_argument(f"--{k}", default=v, type=str) # Force type to str
        elif v_type == list:
            # Handle lists if needed, but attention_resolutions is handled above
            # Assuming other list parameters are not used in this context
            parser.add_argument(f"--{k}", default=v, type=type(v), nargs='+')
        else:
            parser.add_argument(f"--{k}", default=v, type=v_type)

def args_to_dict(args, keys):
    """
    将args对象转换为字典。
    """
    return {k: getattr(args, k) for k in keys}

def str2bool(v):
    """
    将字符串转换为布尔值。
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")