import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login

# Login programmatically
login("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")

# Load the model and tokenizer (using a single model for simplicity)
model_name = "thrishala/mental_health_chatbot"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 for optimized GPU memory
    device_map="auto"  # Automatically map to available GPU/CPU
)

# Initialize the Text Generation Pipeline with the loaded model and tokenizer
mental_health_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.7,  # Lower temperature for more thoughtful responses
    top_p=0.9,
    repetition_penalty=1.2
)

# Define Mental Health Response Function
def generate_mental_health_response(user_input):
    """
    Generates a compassionate and supportive mental health response.
    """
    system_prompt = (
        "You are Connor, a compassionate mental health AI assistant. "
        "Your role is to provide emotional support, active listening, and helpful coping strategies. "
        "Respond empathetically and positively while ensuring users feel heard and supported.\n\n"
    )
    prompt = system_prompt + f"User: {user_input}\nConnor:"
    
    # Generate the AI response
    response = mental_health_pipe(prompt, num_return_sequences=1)[0]['generated_text']
    
    # Extract AI's response after 'Connor:'
    ai_response = response.split("Connor:")[-1].strip()
    return ai_response

# Start the chat with the mental health assistant
if __name__ == "__main__":
    print("🧠 Mental Health AI Assistant (Type 'exit' to quit)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Connor: Take care! Remember, you're not alone. 💖")
            break
        
        ai_reply = generate_mental_health_response(user_input)
        print(f"Connor: {ai_reply}\n")
