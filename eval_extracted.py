import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from sat.model import AutoModel
import torch.nn.functional as F

from utils.utils import llama2_tokenizer, llama2_text_processor_inference, get_image_processor
from utils.models import ExtractRep

def cosine_similarity(tensor1, tensor2):
    tensor1_norm = F.normalize(tensor1, p=2, dim=1)
    tensor2_norm = F.normalize(tensor2, p=2, dim=1)
    return (tensor1_norm * tensor2_norm).sum()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
    parser.add_argument("--chinese", action='store_true', help='Chinese interface')
    parser.add_argument("--version", type=str, default="chat", choices=['chat', 'vqa', 'chat_old', 'base'], help='version of language process. if there is \"text_processor_version\" in model_config.json, this option will be overwritten')
    parser.add_argument("--quant", choices=[8, 4], type=int, default=None, help='quantization bits')

    parser.add_argument("--from_pretrained", type=str, default="cogagent-chat", help='pretrained ckpt')
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--stream_chat", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    args = parser.parse_args()

    # load model
    model, model_args = AutoModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
        deepspeed=None,
        local_rank=rank,
        rank=rank,
        world_size=world_size,
        model_parallel_size=world_size,
        mode='inference',
        skip_init=True,
        use_gpu_initialization=True if (torch.cuda.is_available() and args.quant is None) else False,
        device='cuda',
        **vars(args)
    ), overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {})
    model = model.eval()
    from sat.mpu import get_model_parallel_world_size
    assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()

    # model, model_args = AutoModel.from_pretrained(
    #     args.from_pretrained,
    #     args=args
    # )
    # model = ExtractRep(args)
    # model.eval()

    tokenizer = llama2_tokenizer(args.local_tokenizer)
    image_processor = get_image_processor(model_args.eva_args["image_size"][0])
    
    previous_representation = None

    while True:
        image_path = input("Please enter the image path or URL (type 'stop' to exit): ")
        if image_path.lower() == 'stop':
            break

        processed_image = image_processor(image_path)

        current_representation = model.extract_representations(processed_image.unsqueeze(0))  # Assuming method accepts tensor

        if previous_representation is not None:
            similarity = cosine_similarity(current_representation, previous_representation)
            print(f"Cosine sim: {similarity.item()}")
        else:
            print("Didn't find prev repr")

        previous_representation = current_representation

if __name__ == "__main__":
    main()