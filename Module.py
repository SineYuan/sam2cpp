import torch
from torch import nn
from typing import Any
from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.misc import fill_holes_in_mask_scores

from torchvision.transforms import Normalize, Resize, ToTensor

class ImageEncoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.no_mem_embed = sam_model.no_mem_embed #[1,1,256]
        self.image_encoder = sam_model.image_encoder
        self.num_feature_levels = sam_model.num_feature_levels
        self.prepare_backbone_features = sam_model. _prepare_backbone_features
        self.resolution = 1024
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])


    @torch.no_grad()
    def forward(self, image: torch.Tensor) ->tuple[torch.Tensor, torch.Tensor, torch.Tensor,torch.Tensor,torch.Tensor]:

        image = image.to(torch.float32) / 255.0
        # Apply mean/std normalization
        image = (image - self.mean) / self.std
        # Convert to tensor and permute to CHW
        image = image.permute(2, 0, 1).unsqueeze(0)

        backbone_out = self.image_encoder(image) # {"vision_features","vision_pos_enc","backbone_fpn"}
        backbone_out["backbone_fpn"][0] = self.model.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
        backbone_out["backbone_fpn"][1] = self.model.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])
       
        vision_pos_enc = backbone_out["vision_pos_enc"] # 有3个tensor
        backbone_fpn = backbone_out["backbone_fpn"]     # 有3个tensor
        pix_feat = backbone_out["vision_features"] # 有1个tensor

        expanded_backbone_out = {
            "backbone_fpn": backbone_fpn,
            "vision_pos_enc": vision_pos_enc,
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(1, -1, -1, -1)
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            expanded_backbone_out["vision_pos_enc"][i] = pos.expand(1, -1, -1, -1)
        
        (_,current_vision_feats,current_vision_pos_embeds,_) = self.prepare_backbone_features(expanded_backbone_out)

        current_vision_feat = current_vision_feats[-1] + self.no_mem_embed
        current_vision_feat2 = current_vision_feat.reshape(64,64,1,256).permute(2, 3, 0, 1) # [1,256,64,64]
        
        # flatten HWxNxC -> NxCxHxW
        high_res_features_0 = current_vision_feats[0].reshape(256,256, 1, 32).permute(2, 3, 0, 1) # [1, 32, 256, 256]
        high_res_features_1 = current_vision_feats[1].reshape(128,128, 1, 64).permute(2, 3, 0, 1) # [1, 64, 128, 128]

        # pix_feat              [1, 256, 64, 64]
        # current_vision_feat   [1, 256, 64, 64]
        # current_vision_pos_embed2 [4096, 1, 256]
        # high_res_features_0   [1, 32, 256, 256]
        # high_res_features_1   [1, 64, 128, 128]
        return pix_feat,current_vision_feat2,current_vision_pos_embeds[-1],high_res_features_0,high_res_features_1

class MemAttention0(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.no_mem_embed = sam_model.no_mem_embed
        self.memory_attention = sam_model.memory_attention

    # @torch.no_grad()
    def forward(
        self,
        current_vision_feat: torch.Tensor,      #[1, 256, 64, 64], 当前帧的视觉特征
        current_vision_pos_embed: torch.Tensor, #[4096, 1, 256], 当前帧的位置特征
        memory_0:torch.Tensor,                  # [num_obj_ptr,256]->[num_obj_ptr,4,64]->[4*num_obj_ptr,1,64]
        memory_1:torch.Tensor,                  # [n,64,64,64]->[n,64,4096]->[4096n,1,64]
        memory_pos_embed:torch.Tensor           #[y*4096,1,64], 最近y帧的位置编码特性
    ) -> tuple[Any]:
        num_obj_ptr_tokens =  memory_0.shape[0]*4
        current_vision_feat=current_vision_feat.permute(2,3,0,1).reshape(4096,1,256)
        current_vision_feat = current_vision_feat - self.no_mem_embed

        memory_0 = memory_0.reshape(-1,1,4,64)
        memory_0 = memory_0.permute(0, 2, 1, 3).flatten(0, 1)

        memory_1 = memory_1.view(-1, 64, 64*64).permute(0,2,1)
        memory_1 = memory_1.reshape(-1,1,64)

        print(memory_0.shape,memory_1.shape)
        memory = torch.cat((memory_1,memory_0),dim=0)
        pix_feat_with_mem = self.memory_attention(
            curr = current_vision_feat,
            curr_pos = current_vision_pos_embed,
            memory = memory,
            memory_pos = memory_pos_embed,
            num_obj_ptr_tokens= num_obj_ptr_tokens,
        )
        # reshape the output (HW)xBxC => BxCxHxW
        image_embed = pix_feat_with_mem.permute(1, 2, 0).view(1, 256, 64, 64) # [1,256,64,64]
        return image_embed #[1,256,64,64]

class MemAttention(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.no_mem_embed = sam_model.no_mem_embed
        self.memory_attention = sam_model.memory_attention

    # @torch.no_grad()
    def forward(
        self,
        current_vision_feat: torch.Tensor,      #[1, 256, 64, 64], 当前帧的视觉特征
        current_vision_pos_embed: torch.Tensor, #[4096, 1, 256], 当前帧的位置特征
        maskmem_feats:torch.Tensor,           # [num_maskmem, 4096, 1, 64]
        maskmem_feat_pos_embeds:torch.Tensor, # [num_maskmem, 4096, 1, 64]
        maskmem_feat_pos_idx:torch.Tensor,    # [num_maskmem]
        obj_ptrs:torch.Tensor,  #[num_obj_ptr, 1, 256]
        obj_pos_list:torch.Tensor,  #[num_obj_ptr]
    ) -> tuple[Any]:
        current_vision_feat=current_vision_feat.permute(2,3,0,1).reshape(4096,1,256)
        current_vision_feat = current_vision_feat - self.no_mem_embed

        maskmem_feat_pos_embeds =self.model._prepare_maskmem_feat_pos_embeds(maskmem_feat_pos_idx, maskmem_feat_pos_embeds)
        obj_ptrs, obj_pos = self.model._prepare_obj_ptrs(obj_ptrs, obj_pos_list)

        print(maskmem_feats.shape,obj_ptrs.shape)

        memory = torch.cat((torch.flatten(maskmem_feats, start_dim=0, end_dim=1), obj_ptrs),dim=0)
        memory_pos_embed = torch.cat((torch.flatten(maskmem_feat_pos_embeds, start_dim=0, end_dim=1), obj_pos),dim=0)
        num_obj_ptr_tokens = obj_ptrs.shape[0]

        pix_feat_with_mem = self.memory_attention(
            curr = current_vision_feat,
            curr_pos = current_vision_pos_embed,
            memory = memory,
            memory_pos = memory_pos_embed,
            num_obj_ptr_tokens= num_obj_ptr_tokens,
        )
        # reshape the output (HW)xBxC => BxCxHxW
        image_embed = pix_feat_with_mem.permute(1, 2, 0).view(1, 256, 64, 64) # [1,256,64,64]
        return image_embed #[1,256,64,64]

class MemEncoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.maskmem_tpos_enc = sam_model.maskmem_tpos_enc
        self.feat_sizes = [(256, 256), (128, 128), (64, 64)]
    @torch.no_grad()
    def forward(
        self,
        mask_for_mem: torch.Tensor,  # [1,1,1024,1024]
        pix_feat: torch.Tensor,      # [1,256,64,64]
        object_score_logits: torch.Tensor,       #[1,1]
    )-> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
            current_vision_feats=pix_feat,
            #feat_sizes=self.feat_sizes,
            feat_sizes=None,
            pred_masks_high_res=mask_for_mem,
            is_mask_from_pts=True,
            object_score_logits=object_score_logits,
        )
        print(maskmem_features.shape)
        #maskmem_features = maskmem_features.view(1, 64, 64*64).permute(2, 0, 1)
        #maskmem_pos_enc = maskmem_pos_enc[0].view(1, 64, 64*64).permute(2, 0, 1)
        maskmem_features = maskmem_features.flatten(2).permute(2, 0, 1)
        maskmem_pos_enc = maskmem_pos_enc[0].flatten(2).permute(2, 0, 1)

        #return maskmem_features,maskmem_pos_enc,self.maskmem_tpos_enc
        return maskmem_features,maskmem_pos_enc

class ImageDecoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.sigmoid_scale_for_mem_enc = sam_model.sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sam_model.sigmoid_bias_for_mem_enc
    @torch.no_grad()
    def forward(
        self,
        point_coords: torch.Tensor, # [num_labels,num_points,2]
        point_labels: torch.Tensor, # [num_labels,num_points]
        #frame_size: torch.Tensor,   # [2]
        image_height: torch.Tensor,   # []
        image_width: torch.Tensor,   # []
        image_embed: torch.Tensor,  # [1,256,64,64]
        high_res_feats_0: torch.Tensor, # [1, 32, 256, 256]
        high_res_feats_1: torch.Tensor, # [1, 64, 128, 128]
    ):
        point_inputs = {"point_coords":point_coords,"point_labels":point_labels}
        high_res_feats = [high_res_feats_0, high_res_feats_1]

        sam_outputs = self.model._forward_sam_heads(
            backbone_features=image_embed,
            point_inputs=point_inputs,
            mask_inputs=None,
            high_res_features=high_res_feats,
            multimask_output=False,
        )
        (
            _,
            _,
            ious,
            low_res_masks, # [1,1,256,256]
            high_res_masks, # [1,1,1024,1024]
            obj_ptr,  # [1,256]
            object_score_logits, # [1, 1]
        ) = sam_outputs
        # 处理高分辨率mask
        mask_for_mem = torch.sigmoid(high_res_masks)
        mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        # 填洞
        #low_res_masks = fill_holes_in_mask_scores(low_res_masks, 8)
        # 还原到原图大小
        pred_mask = torch.nn.functional.interpolate(
            low_res_masks,
            #size=(frame_size[0], frame_size[1]),
            size=(image_height, image_width),
            mode="bilinear",
            align_corners=False,
        )
        return obj_ptr,mask_for_mem,pred_mask,ious,object_score_logits