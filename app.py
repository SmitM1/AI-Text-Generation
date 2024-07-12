import gradio as gr
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large')

def generate_blog_post(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=500, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text

# Set up the Gradio interface
iface = gr.Interface(
    fn=generate_blog_post,
    inputs="text",
    outputs="text",
    title="Blog Post Generator",
    description="Enter a prompt to generate a blog post using GPT-2."
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()
