from transformers import WhisperTokenizer
model_name = "openai/whisper-large-v2"
normalizer = WhisperTokenizer.from_pretrained(model_name, task="transcribe")

def normalize_text(transcript: str, normalizer = normalizer) -> str:
    transcript = transcript.replace("(", "").replace(")", "")
    normalized_text = normalizer.normalize(transcript)
    cleaned_text = " ".join(normalized_text.split())
    return cleaned_text

