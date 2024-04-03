import argparse
import torch
import time
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

import logging
import h5py
import os
import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_cosine_similarity(tensor1, tensor2):
    tensor1_flat = tensor1.view(tensor1.size(0), -1)
    tensor2_flat = tensor2.view(tensor2.size(0), -1)

    similarity = torch.nn.functional.cosine_similarity(tensor1_flat, tensor2_flat, dim=1)
    return similarity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", choices=[4], type=int, default=None, help="Quantization bits")
    parser.add_argument("--image_folder", type=str, help="Folder containing images")
    parser.add_argument("--from_pretrained", type=str, default="THUDM/cogagent-chat-hf", help="Pretrained model checkpoint")
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help="Tokenizer path")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    args = parser.parse_args()

    command_list_txt = [
        # "Describe the scene",
        # "The position of red block with respect to blue block is",
        # "The position of red block with respect to the white box is"
        # "The distance between robot gripper and red block is",
        "The distance between red block and white box is",
    ]

    for img_name in sorted(os.listdir(args.image_folder)):
        image_path = os.path.join(args.image_folder, img_name)
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue

        print(f"\033[1;31mProcessing {img_name}..." + "\033[0m")

        # Load the image
        image = Image.open(image_path).convert('RGB')

        MODEL_PATH = args.from_pretrained
        TOKENIZER_PATH = args.local_tokenizer
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
        torch_type = torch.float16 # on A100, use torch.bfloat16

        logger.info("\033[1;31mSTARTING " + "\033[0m")
        logger.info("========Use torch type as:{} with device:{}========\n\n".format(torch_type, DEVICE))

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch_type,
            low_cpu_mem_usage=True,
            load_in_4bit=args.quant is not None,
            trust_remote_code=True
        ).to(DEVICE).eval()

        is_log = True

        base_folder_path = "../logs/logs_vid_1131"
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"N33_{img_name}_{current_time}"
        run_dir_path = os.path.join(base_folder_path, current_date, run_name)
        if is_log:
            os.makedirs(run_dir_path, exist_ok=True)
        log_file_path = os.path.join(run_dir_path, "run_log.log")

        image = Image.open(image_path).convert('RGB')
        history = []

        # Process each prompt for the current image
        for query in command_list_txt:
            input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=history, images=[image])

            inputs = {
                'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
                'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
                'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
                'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]] if image is not None else None,
            }
            if 'cross_images' in input_by_model and input_by_model['cross_images']:
                inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch_type)]]

            gen_kwargs = {"max_length": 2048,
                            "do_sample": False} # "temperature": 0.9
            with torch.no_grad():
                model_outputs = model(**inputs, output_hidden_states=True, output_attentions=True, extract_intermediate_representation=True, return_dict=True)
                # import ipdb; ipdb.set_trace()
                encoder_hidden_states = model_outputs.hidden_states
                # print(dir(model_outputs))
                logger.info(f"\033[1;31m{query} " + "\033[0m")
                # import ipdb; ipdb.set_trace()
                print(f"Hidden states shape: {encoder_hidden_states[0].shape}")
                # log hidden states to hdf5
                if is_log:
                    with h5py.File(os.path.join(run_dir_path, 'hidden_states.h5'), 'w') as f:
                        f.create_dataset('hidden_states', data=encoder_hidden_states[0].cpu().numpy())
                intermediate_representation = model_outputs.intermediate_representations
                print("Intermediate Representations:")
                print(intermediate_representation.keys())
                print(intermediate_representation['self_attn_weights'][-1].shape)
                print(intermediate_representation['cross_attn_weights'][-1].shape)
                print(intermediate_representation['hidden_states'][-1].shape)
                #log intermediate representations dict to hdf5
                logger.info(f"Log file path: {log_file_path}")
                if is_log:
                    with h5py.File(os.path.join(run_dir_path, f'{img_name}_intermediate_representations.h5'), 'w') as f:
                        for key, value in intermediate_representation.items():
                            f.create_dataset(key, data=value[-1].cpu().numpy())

                # if model_outputs.attentions is not None:
                #     encoder_attentions = model_outputs.attentions
                #     print(f"Attention shape: {encoder_attentions[-1].shape}")
                # else:
                #     print("Attention weights are not available.")
                intr_outputs = model.generate(**inputs, **gen_kwargs)
                # print("Methods in Model:", dir(model))
                # test_outputs = model.get_output_embeddings()
                # weight_shape = test_outputs.weight.shape
                # print("Output Embeddings Shape:", weight_shape)
                #log output embeddings to hdf5
                # with h5py.File(os.path.join(run_dir_path, 'output_embeddings.h5'), 'w') as f:
                #     f.create_dataset('output_embeddings', data=test_outputs.weight.cpu().numpy())
                reply_outputs = intr_outputs[:, inputs['input_ids'].shape[1]:]
                response = tokenizer.decode(reply_outputs[0], skip_special_tokens=True)
                response = response.split("</s>")[0]
                print("Shapes of Inputs: input_ids", inputs['input_ids'].shape, "token_type_ids", inputs['token_type_ids'].shape, "attention_mask", inputs['attention_mask'].shape)
                print("Shapes of Intrinsic Outputs:", intr_outputs.shape)
                # print("Intrinsic Output:", intr_outputs)
                print("Number of Non-Zero Items in Intrinsic Outputs:", (intr_outputs != 0).sum().item())
                print("Shape of Reply Outputs:", reply_outputs.shape)
                # print("Response:", response)

            history.append((query, response))

        print(f"Finished processing {img_name}.")
        del model
        time.sleep(2)   

if __name__ == "__main__":
    main()
