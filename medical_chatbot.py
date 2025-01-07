import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import logging
import warnings
warnings.filterwarnings('ignore')

class MedicalChatbot:
    def __init__(self, base_model_path="meta-llama/Llama-3.2-1B", adapter_path="./llama3-mental-health-counselor-final"):
        """
        Initialize the medical chatbot with the fine-tuned model
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                device_map={"": 0}
            )
            
            # Load base model
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=bnb_config,
                device_map={"": 0},
                torch_dtype=torch.float16
            )
            
            # Load adapter weights
            self.model = PeftModel.from_pretrained(
                self.base_model,
                adapter_path
            )
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            raise

    def generate_response(self, user_input, max_length=512, temperature=0.7):
        """
        Generate a response to the user's input
        """
        try:
            # Format the input
            prompt = f"### Instruction: Act as a mental health counselor. Respond to the following message:\n\n{user_input}\n\n### Response:"
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to("cuda")
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    repetition_penalty=1.2
                )
            
            # Decode and clean response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("### Response:")[-1].strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error generating a response. Please try again."

    def chat(self):
        """
        Interactive chat loop
        """
        print("Medical Chatbot initialized. Type 'quit' to exit.")
        print("Note: This is an AI assistant and not a replacement for professional medical advice.\n")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nChatbot: Take care! Goodbye!")
                break
                
            if not user_input:
                print("Chatbot: Please type something.")
                continue
                
            response = self.generate_response(user_input)
            print(f"\nChatbot: {response}")

def main():
    try:
        # Create chatbot instance
        chatbot = MedicalChatbot()
        
        # Start chat interface
        chatbot.chat()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please ensure you have the correct model paths and required dependencies installed.")

if __name__ == "__main__":
    main()