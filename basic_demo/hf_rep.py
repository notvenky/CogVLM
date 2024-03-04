import argparse
import torch

from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

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
torch_type = torch.float16 # on A100, use torch.bfloat16


print("\033[1;31m#$#$ " * 30 + "\033[0m")
print("========Use torch type as:{} with device:{}========\n\n".format(torch_type, DEVICE))

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch_type,
    low_cpu_mem_usage=True,
    load_in_4bit=args.quant is not None,
    trust_remote_code=True
).to(DEVICE).eval()

image_path = "../../Downloads/vlm/IMG_0864.jpg"
command_list_txt = ["What is the distance between tennis ball and bottle of disinfectant"] #, "Where is the tennis ball and bottle of disinfectant"]

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

    gen_kwargs = {"max_length": 2048,
                    "do_sample": False} # "temperature": 0.9
    with torch.no_grad():
        model_outputs = model(**inputs, output_hidden_states=True, output_attentions=True, extract_intermediate_representation=True, return_dict=True)
        encoder_hidden_states = model_outputs.hidden_states
        # intermediate_representation = model_outputs.intermediate_representations
        # print(dir(model_outputs))
        print(f"Hidden states shape: {encoder_hidden_states[-1].shape}")
        intermediate_representation = model_outputs.intermediate_representations
        print(intermediate_representation)
        


        # if model_outputs.attentions is not None:
        #     encoder_attentions = model_outputs.attentions
        #     print(f"Attention shape: {encoder_attentions[-1].shape}")
        # else:
        #     print("Attention weights are not available.")
        intr_outputs = model.generate(**inputs, **gen_kwargs)
        # print("Methods in Model:", dir(model))
        test_outputs = model.get_output_embeddings()
        weight_shape = test_outputs.weight.shape
        # print('Output Embeddings:', test_outputs.weight)
        # print("Output Embeddings Shape:", weight_shape)
        reply_outputs = intr_outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(reply_outputs[0])
        response = response.split("</s>")[0]

        print("Query:", query)
        # print("Shapes of Inputs: input_ids", inputs['input_ids'].shape, "token_type_ids", inputs['token_type_ids'].shape, "attention_mask", inputs['attention_mask'].shape)
        print("Shapes of Intrinsic Outputs:", intr_outputs.shape)
        # print("Intrinsic Output:", intr_outputs)
        print("Number of Non-Zero Items:", (intr_outputs != 0).sum().item())
        print("Shape of Reply Outputs:", reply_outputs.shape)
        print("Response:", response)

    history.append((query, response))