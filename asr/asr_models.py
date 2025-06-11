import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import pipeline
from fireredasr.models.fireredasr import FireRedAsr

class WhisperASR:
    def __init__(self, model_name="openai/whisper-large-v2", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.processor = WhisperProcessor.from_pretrained(model_name, task="transcribe")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def transcribe_audio(self, audio_path, language="en"):

        audio, sr = torchaudio.load(audio_path)
        audio = audio.squeeze(0)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            audio = resampler(audio)

        with torch.no_grad():
            whisper_inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", truncation=False, padding=True)
            whisper_inputs = {k: v.to(self.device) for k, v in whisper_inputs.items()}
            # attention_mask is add after see the warning
            attention_mask = (whisper_inputs['input_features'] != self.processor.tokenizer.pad_token_id).long()
            whisper_inputs['attention_mask'] = attention_mask
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")

            asr_model_out = self.model.generate(
                **whisper_inputs, 
                forced_decoder_ids=forced_decoder_ids,
                return_timestamps="word",
                return_segments=True
            )

            transcripts = self.processor.batch_decode(asr_model_out['sequences'], output_offsets=True, skip_special_tokens=True)

        return transcripts[0]['text'].strip()

class FireRedASR_AED_L_ASRModel:
    def __init__(self, model_name="aed", model_path="./FireRedASR/pretrained_models/FireRedASR-AED-L"):
        self.model = FireRedAsr.from_pretrained(model_name, model_path)

    def transcribe_audio(self, audio_path, language="zh"):
        with torch.no_grad():
            results = self.model.transcribe(
                ["dummy_id"],
                [audio_path],
                {
                    "use_gpu": 1,
                    "beam_size": 3,
                    "nbest": 1,
                    "decode_max_len": 0,
                    "softmax_smoothing": 1.0,
                    "aed_length_penalty": 0.0,
                    "eos_penalty": 1.0
                }
            )
            transcription = results[0]['text']
            return transcription.strip()
        
# if __name__ == "__main__":
#     # Example usage
#     model = WhisperASR()
#     audio_file = "/root/shared-nvme/datasets/Fisher/eval/16k/fe_03_00927.wav"
#     transcription = model.transcribe_audio(audio_file)
#     print("Transcription:", transcription)