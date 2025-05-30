import torch
import onnx
import argparse
from onnxsim import simplify
from Module import ImageEncoder, ImageDecoder, MemAttention, MemEncoder
from sam2.build_sam import build_sam2

def export_image_encoder(model,onnx_path):
    #input_img = torch.randn(1, 3,1024, 1024).cpu()
    input_img = torch.randn(1024, 1024, 3).cpu()
    # transform to uint8 map (0, 1) to [0,255]
    input_img = (input_img * 255).to(torch.uint8)
    output_names = ["pix_feat","vision_feats","vision_pos_embed","high_res_feat0","high_res_feat1"]
    #output_names = ["pix_feat","high_res_feat0","high_res_feat1","current_vision_feat2","current_vision_pos_embed"]
    torch.onnx.export(
        model,
        input_img,
        onnx_path+"image_encoder.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=output_names,
    )
    # # 简化模型, tmd将我的输出数量都简化掉一个，sb
    # original_model = onnx.load(onnx_path+"image_encoder.onnx")
    # simplified_model, check = simplify(original_model)
    # onnx.save(simplified_model, onnx_path+"image_encoder.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"image_encoder.onnx")
    onnx.checker.check_model(onnx_model)
    print("image_encoder.onnx model is valid!")
    


def export_memory_attention(model,onnx_path):
    current_vision_feat = torch.randn(1,256,64,64)      #[1, 256, 64, 64],当前帧的视觉特征
    current_vision_pos_embed = torch.randn(4096,1,256)  #[4096, 1, 256],当前帧的位置特征

    maskmem_feats = torch.randn(7,4096,1,64)            # [num_maskmem, 4096, 1, 64]
    maskmem_feat_pos_embeds = torch.randn(7,4096,1,64)  # [num_maskmem, 4096, 1, 64]
    maskmem_feat_pos_idx = torch.tensor([6, 6, 5, 3, 2, 1, 0], dtype=torch.int32)               # [num_maskmem]
    obj_ptrs = torch.randn(8,1,256)                    # [num_obj_ptr, 1, 256]
    obj_pos_list = torch.tensor([8, 5, 1, 2, 3, 4, 6, 7], dtype=torch.int32)                      # [num_obj_ptr]

    out = model(
            current_vision_feat = current_vision_feat,
            current_vision_pos_embed = current_vision_pos_embed,
            maskmem_feats = maskmem_feats,
            maskmem_feat_pos_embeds = maskmem_feat_pos_embeds,
            maskmem_feat_pos_idx = maskmem_feat_pos_idx,
            obj_ptrs = obj_ptrs,
            obj_pos_list = obj_pos_list
        )
    input_name = ["current_vision_feat",
                "current_vision_pos_embed",
                "maskmem_feats",
                "maskmem_feat_pos_embeds",
                "maskmem_feat_pos_idx",
                "obj_ptrs",
                "obj_pos_list"]
    dynamic_axes = {
        "maskmem_feats": {0: "num_maskmem"},
        "maskmem_feat_pos_embeds": {0: "num_maskmem"},
        "maskmem_feat_pos_idx": {0: "num_maskmem"},
        "obj_ptrs": {0: "num_obj_ptr"},
        "obj_pos_list": {0: "num_obj_ptr"}
    }
    print("going to export memory_attention.onnx")
    torch.onnx.export(
        model,
        (current_vision_feat,current_vision_pos_embed,maskmem_feats,maskmem_feat_pos_embeds,maskmem_feat_pos_idx,obj_ptrs,obj_pos_list),
        onnx_path+"memory_attention.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=["image_embed"],
        dynamic_axes = dynamic_axes
    )
     # 简化模型,
    original_model = onnx.load(onnx_path+"memory_attention.onnx")
    simplified_model, check = simplify(original_model)
    onnx.save(simplified_model, onnx_path+"memory_attention.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"memory_attention.onnx")
    onnx.checker.check_model(onnx_model)
    print("memory_attention.onnx model is valid!")

def export_image_decoder(model,onnx_path):
    point_coords = torch.randn(1,2,2).cpu()
    point_labels = torch.randn(1,2).cpu()
    frame_size = torch.tensor([1024,1024],dtype=torch.int64)
    image_height = torch.tensor(1024,dtype=torch.int32)
    image_width = torch.tensor(1024,dtype=torch.int32)
    image_embed = torch.randn(1,256,64,64).cpu()
    high_res_feats_0 = torch.randn(1,32,256,256).cpu()
    high_res_feats_1 = torch.randn(1,64,128,128).cpu()

    out = model(
        point_coords = point_coords,
        point_labels = point_labels,
        #frame_size = frame_size,
        image_height = image_height,
        image_width = image_width,
        image_embed = image_embed,
        high_res_feats_0 = high_res_feats_0,
        high_res_feats_1 = high_res_feats_1
    )
    #input_name = ["point_coords","point_labels","frame_size","image_embed","high_res_feats_0","high_res_feats_1"]
    input_name = ["point_coords","point_labels","image_height", "image_width", "image_embed","high_res_feats_0","high_res_feats_1"]
    output_name = ["obj_ptr","mask_for_mem","pred_mask","ious","object_score_logits"]
    dynamic_axes = {
        #"point_coords":{0: "num_labels",1:"num_points"},
        #"point_labels": {0: "num_labels",1:"num_points"}
        "point_coords":{1:"num_points"},
        "point_labels": {1:"num_points"}
    }
    torch.onnx.export(
        model,
        #(point_coords,point_labels,frame_size,image_embed,high_res_feats_0,high_res_feats_1),
        (point_coords,point_labels,image_height,image_width,image_embed,high_res_feats_0,high_res_feats_1),
        onnx_path+"image_decoder.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=output_name,
        dynamic_axes = dynamic_axes
    )
    # 简化模型,
    original_model = onnx.load(onnx_path+"image_decoder.onnx")
    simplified_model, check = simplify(original_model)
    onnx.save(simplified_model, onnx_path+"image_decoder.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"image_decoder.onnx")
    onnx.checker.check_model(onnx_model)
    print("image_decoder.onnx model is valid!")

def export_memory_encoder(model,onnx_path):
    mask_for_mem = torch.randn(1,1,1024,1024) 
    pix_feat = torch.randn(1,256,64,64) 
    object_score_logits = torch.randn(1,1)
    out = model(mask_for_mem = mask_for_mem,pix_feat = pix_feat, object_score_logits = object_score_logits)

    input_names = ["mask_for_mem","pix_feat","object_score_logits"]
    #output_names = ["maskmem_features","maskmem_pos_enc","temporal_code"]
    output_names = ["maskmem_features","maskmem_pos_enc"]
    torch.onnx.export(
        model,
        (mask_for_mem,pix_feat,object_score_logits),
        onnx_path+"memory_encoder.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_names,
        output_names= output_names
    )
    # 简化模型,
    # original_model = onnx.load(onnx_path+"memory_encoder.onnx")
    # simplified_model, check = simplify(original_model)
    # onnx.save(simplified_model, onnx_path+"memory_encoder.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"memory_encoder.onnx")
    onnx.checker.check_model(onnx_model)
    print("memory_encoder.onnx model is valid!")

#****************************************************************************
model_type = ["tiny","small","large","base+"][3]
onnx_output_path = "checkpoints/{}/".format(model_type)
model_config_file = "sam2_hiera_{}.yaml".format(model_type)
model_checkpoints_file = "checkpoints/sam2_hiera_{}.pt".format(model_type)

onnx_output_path = "checkpoints/{}/".format("small")
model_checkpoints_file = "/home/stargi/learnzone/sam2/checkpoints/sam2.1_hiera_small.pt"
model_config_file = "configs/sam2.1/sam2.1_hiera_s.yaml"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出SAM2为onnx文件")
    parser.add_argument("--outdir",type=str,default=onnx_output_path,required=False,help="path")
    parser.add_argument("--config",type=str,default=model_config_file,required=False,help="*.yaml")
    parser.add_argument("--checkpoint",type=str,default=model_checkpoints_file,required=False,help="*.pt")
    args = parser.parse_args()
    sam2_model = build_sam2(args.config, args.checkpoint, device="cpu")

    image_encoder = ImageEncoder(sam2_model).cpu()
    export_image_encoder(image_encoder,args.outdir)

    image_decoder = ImageDecoder(sam2_model).cpu()
    export_image_decoder(image_decoder,args.outdir)

    mem_encoder   = MemEncoder(sam2_model).cpu()
    export_memory_encoder(mem_encoder,args.outdir)

    mem_attention = MemAttention(sam2_model).cpu()
    export_memory_attention(mem_attention,args.outdir)

    print("the end")