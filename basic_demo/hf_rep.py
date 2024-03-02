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


print("\033[1;31m#$#$ " * 40 + "\033[0m")
print("========Use torch type as:{} with device:{}========\n\n".format(torch_type, DEVICE))

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch_type,
    low_cpu_mem_usage=True,
    load_in_4bit=args.quant is not None,
    trust_remote_code=True
).to(DEVICE).eval()

image_path = "../../Downloads/vlm/IMG_0864.jpg"
command_list_txt = ["Describe the scene", "What is the distance between tennis ball and bottle of disinfectant", "Where is the tennis ball and bottle of disinfectant"]
# command_list_txt = ["What is the distance between tennis ball and bottle of disinfectant"]
# command_list_txt = ["Where is the tennis ball and bottle of disinfectant"]

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
        intr_outputs = model.generate(**inputs, **gen_kwargs)
        print("Methods in Model:", dir(model))
        test_outputs = model.get_output_embeddings()
        weight_shape = test_outputs.weight.shape
        print("Output Embeddings Shape:", weight_shape)
        reply_outputs = intr_outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(reply_outputs[0])
        response = response.split("</s>")[0]

        print("Query:", query)
        print("Image Shape:", inputs['images'][0][0].shape if inputs['images'] else None)
        print("Cross Image Shape:", inputs['cross_images'][0][0].shape if inputs['cross_images'] else None)
        print("Shapes of Inputs: input_ids", inputs['input_ids'].shape, "token_type_ids", inputs['token_type_ids'].shape, "attention_mask", inputs['attention_mask'].shape)
        print("Shapes of Intrinsic Outputs:", intr_outputs.shape)
        # print("Intrinsic Output:", intr_outputs)
        print("Number of Non-Zero Items:", (intr_outputs != 0).sum().item())
        print("Shape of Reply Outputs:", reply_outputs.shape)
        # print("Reply Outputs:", reply_outputs)
        print("Response:", response)

    history.append((query, response))


"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class CustomModelForCausalLMWithStates(AutoModelForCausalLM):
    def generate_with_states(self, input_ids, attention_mask=None, **kwargs):
        # Ensure the model is in evaluation mode
        self.eval()

        # Initialize an empty list to store the states
        all_states = []

        # Custom generate loop to capture internal states
        # Note: This is a simplified logic for demonstration. Actual implementation may vary.
        with torch.no_grad():
            # Assuming 'input_ids' is already prepared for the model
            outputs = self(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            # Extract the last hidden state (for simplicity)
            last_hidden_state = outputs.hidden_states[-1]
            all_states.append(last_hidden_state)

            # For actual text generation, you might need to manually implement logic
            # similar to what `generate` does, or modify this to call `generate`
            # and capture states in each step. This example focuses on extracting states.
            
            # Placeholder for generated token IDs (you would implement generation logic here)
            generated_token_ids = torch.tensor([[0]])  # Dummy tensor, replace with actual generation logic

        return generated_token_ids, all_states

# Example usage
MODEL_PATH = "gpt2"  # Example model, replace with your model path
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = CustomModelForCausalLMWithStates.from_pretrained(MODEL_PATH)

# Prepare inputs
input_text = "The quick brown fox jumps over the lazy dog"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
attention_mask = torch.ones(input_ids.shape, dtype=torch.long)  # Assuming full attention

# Generate text and capture states
generated_token_ids, all_states = model.generate_with_states(input_ids, attention_mask)

# Decode generated text (for demonstration)
generated_text = tokenizer.decode(generated_token_ids[0], skip_special_tokens=True)
print("Generated Text:", generated_text)
    
"""