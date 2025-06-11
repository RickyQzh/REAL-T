from transformers import WhisperTokenizer
import opencc
import re

model_name = "openai/whisper-large-v2"
normalizer = WhisperTokenizer.from_pretrained(model_name)

def whisper_normalize(transcript: str) -> str:
    transcript = transcript.replace("(", "").replace(")", "")
    normalized_text = normalizer.normalize(transcript)
    cleaned_text = " ".join(normalized_text.split())
    return cleaned_text.strip()


def normalizer_for_zh(transcript: str, option: str = None) -> str:

    assert option in ['Predicted', 'Ground Truth'], f"Invalid option: {option}"
    if option == 'Predicted':
        transcript = whisper_normalize(transcript)
    elif option == 'Ground Truth':
        pass

    # transcript = whisper_normalize(transcript.strip()).strip()
    converter = opencc.OpenCC('t2s')
    transcript = converter.convert(transcript)
    transcript = re.sub(r'\s+', '', transcript)
    transcript = " ".join(transcript.strip())
    return transcript

def normalizer_for_en(transcript: str, option: str = None) -> str:

    transcript = transcript.strip()
    transcript = transcript.replace("…", "")
    assert option in ['Predicted', 'Ground Truth'], f"Invalid option: {option}"
    if option == 'Predicted':
        transcript = whisper_normalize(transcript)
    elif option == 'Ground Truth':
        pass
    
    transcript = transcript.replace(".", "")
    re.sub(r'\s+', ' ', transcript.strip())
    return transcript.strip()

def normalizer_for_transcript(transcript: str, option: str = None, language: str = None) -> str:

    assert language in ['zh', 'en']

    if language == 'en':
        transcript = normalizer_for_en(transcript, option)
    elif language == 'zh':
        transcript = normalizer_for_zh(transcript, option)
    
    return transcript.strip()
