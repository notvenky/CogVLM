import argparse
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
import logging
import h5py
import os
from datetime import datetime

base_folder_path = "../logs"
current_date = datetime.now().strftime("%Y-%m-%d")
current_time = datetime.now().strftime("%H-%M-%S")
run_name = f"run_{current_time}"
run_dir_path = os.path.join(base_folder_path, current_date, run_name)
os.makedirs(run_dir_path, exist_ok=True)

log_file_path = os.path.join(run_dir_path, "run_log.log")
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--quant", choices=[4], type=int, default=None, help='quantization bits')
parser.add_argument("--from_pretrained", type=str, default="THUDM/cogagent-chat-hf", help='pretrained ckpt')
parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
parser.add_argument("--fp16", action="store_true")
args = parser.parse_args()

MODEL_PATH = args.from_pretrained
TOKENIZER_PATH = args.local_tokenizer
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
torch_type = torch.float16

logger.info("Use torch type as:{} with device:{}".format(torch_type, DEVICE))

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch_type, low_cpu_mem_usage=True, load_in_4bit=args.quant is not None, trust_remote_code=True).to(DEVICE).eval()

image_path = "../../Downloads/vlm/IMG_0864.jpg"
command_list_txt = ["What is the distance between tennis ball and bottle of disinfectant"]

image = Image.open(image_path).convert('RGB')
history = []

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

    gen_kwargs = {"max_length": 2048, "do_sample": False}
    with torch.no_grad():
        model_outputs = model(**inputs, output_hidden_states=True, output_attentions=True, extract_intermediate_representation=True, return_dict=True)
        encoder_hidden_states = model_outputs.hidden_states
        intermediate_representation = model_outputs.intermediate_representations
        logger.info(f"Hidden states shape: {encoder_hidden_states[-1].shape}")
        logger.info("Intermediate Representations: {}".format(list(intermediate_representation.keys())))
        
        hdf5_path = os.path.join(run_dir_path, f"features_responses_{current_time}.h5")
        with h5py.File(hdf5_path, 'w') as hf:
            for layer_name, metrics in intermediate_representation.items():
                if 'self_attn_weights' in metrics:
                    for idx, weight in enumerate(metrics['self_attn_weights']):
                        hf.create_dataset(f"{layer_name}/self_attn_weights_{idx}", data=weight.cpu().numpy())
                if 'cross_attn_weights' in metrics:
                    for idx, weight in enumerate(metrics['cross_attn_weights']):
                        hf.create_dataset(f"{layer_name}/cross_attn_weights_{idx}", data=weight.cpu().numpy())

        intr_outputs = model.generate(**inputs, **gen_kwargs)
        reply_outputs = intr_outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(reply_outputs[0], skip_special_tokens=True).split("</s>")[0]
        logger.info("Query: {}".format(query))
        logger.info("Shapes of Intrinsic Outputs: {}".format(intr_outputs.shape))
        logger.info("Number of Non-Zero Items: {}".format((intr_outputs != 0).sum().item()))
        logger.info("Shape of Reply Outputs: {}".format(reply_outputs.shape))
        logger.info("Response: {}".format(response))
    history.append((query, response))
