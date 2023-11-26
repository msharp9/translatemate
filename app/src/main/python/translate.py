from os.path import dirname, join
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, NllbTokenizerFast, M2M100ForConditionalGeneration, pipeline


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


path = join(dirname(__file__), "nllb-200-distilled-600M")
tokenizer = NllbTokenizerFast.from_pretrained(path)
nllb_model = M2M100ForConditionalGeneration.from_pretrained(path)
nllb_model = nllb_model.to_bettertransformer()


path = join(dirname(__file__), "distil-whisper")
processor = WhisperProcessor.from_pretrained(path)
whisper_model = WhisperForConditionalGeneration.from_pretrained(
    path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
whisper_model = whisper_model.to_bettertransformer()
whisper_model.to(device)

pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15,
    batch_size=16,
    torch_dtype=torch_dtype,
    device=device,
)

def translate_english_audio_to_chinese_text(path: str) -> str:
    audio_result = pipe(path)
    inputs = tokenizer(audio_result["text"], return_tensors="pt")
    translated_tokens = nllb_model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["zho_Hant"]
    )
    results = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return results
