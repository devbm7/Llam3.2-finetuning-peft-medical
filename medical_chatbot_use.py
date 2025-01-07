# Basic usage with interactive chat
from medical_chatbot import MedicalChatbot

chatbot = MedicalChatbot()
# chatbot.chat()

# Or use it programmatically
response = chatbot.generate_response("What should i do about headaches?")
print(response)