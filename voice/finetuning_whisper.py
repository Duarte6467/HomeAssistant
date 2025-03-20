from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
from dataclasses import dataclass
from typing import Any,Dict,List,Union
import evaluate

if torch.cuda.is_available():
    print("CUDA is available. Running on GPU.")
    torch.backends.cudnn.benchmark = True
else:
    print("CUDA is not available. Running on CPU.")

common_voice = DatasetDict()
common_voice["train"] = load_dataset("mozilla-foundation/common_voice_17_0", "pt", split="train", trust_remote_code=True)
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_17_0", "pt", split="test", trust_remote_code=True)

common_voice["train"] = common_voice["train"].filter(lambda example: example['variant'] != 'Portuguese(Brazil)')
common_voice["test"] = common_voice["test"].filter(lambda example: example['variant'] != 'Portuguese(Brazil)')

common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes","variant"])

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", language="pt", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="pt", task="transcribe")
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"])


model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
model.generation_config.language = "pt"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None



@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyway
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch



data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)


metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == 100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-hi",  # change to a repo name of your choice
    per_device_train_batch_size=32,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=10000,
    gradient_checkpointing=False,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    hub_private_repo=True,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model.to("cuda"),
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)



trainer.train(resume_from_checkpoint=True)


kwargs = {
    "dataset_tags": "mozilla-foundation/common_voice_17_0",
    "dataset": "Common Voice 17.0",  # a 'pretty' name for the training dataset
    "dataset_args": "config: pt, split: test",
    "language": "pt",
    "model_name": "Whisper-base-pt-pt",  # a 'pretty' name for your model
    "finetuned_from": "openai/whisper-base",
    "tasks": "automatic-speech-recognition",
}


trainer.push_to_hub(**kwargs)