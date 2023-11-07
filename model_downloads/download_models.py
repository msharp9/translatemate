# Just download and save
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

print("Loading models...")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

processor = AutoProcessor.from_pretrained("distil-whisper/distil-large-v2")
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained("distil-whisper/distil-large-v2")

print("Saving models...")
tokenizer.save_pretrained("./app/src/main/python/nllb-200-distilled-600M")
nllb_model.save_pretrained("./app/src/main/python/nllb-200-distilled-600M")

processor.save_pretrained("./app/src/main/python/distil-whisper")
whisper_model.save_pretrained("./app/src/main/python/distil-whisper")

print("Done")