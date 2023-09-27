from transformers import MarianTokenizer, MarianMTModel

# Load the MarianMT model and tokenizer for English to Hindi
model_name = "Helsinki-NLP/opus-mt-en-hi"  # English to Hindi
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Define a function for translation
def translate_to_hinglish(english_text):
    # Tokenize and translate the text
    inputs = tokenizer.encode(">>en-hi<< " + english_text, return_tensors="pt")
    translated = model.generate(inputs, max_length=150, num_beams=5, early_stopping=True)

    # Decode and return the translated text
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Example English text statements
english_statements = [
    "Hello, how are you?",
    "I am learning NLP with transformers.",
    "This is a simplified translation example."
]

# Translate each statement to Hinglish
for statement in english_statements:
    hinglish_translation = translate_to_hinglish(statement)
    print(f"English: {statement}")
    print(f"Hinglish: {hinglish_translation}\n")
