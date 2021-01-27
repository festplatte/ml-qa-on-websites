from transformers import pipeline
from transformers import Trainer, TrainingArguments, AutoModelWithLMHead
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import AutoTokenizer

train_path = '/data/ms-marco/train_dataset.txt'
test_path = '/data/ms-marco/dev_dataset.txt'
cache_dir = '/data/.cache'
output_path  = '/data/gpt2-ms-marco'

# load tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained(
    "gpt2", cache_dir=cache_dir, additional_special_tokens=['<SOC>', '<SOQ>', '<SOA>'], eos_token="<EOS>", pad_token="<PAD>", bos_token="<BOS>")
print('tokenizer loaded')


def load_dataset(train_path, test_path, tokenizer):
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=128)

    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=test_path,
        block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset, test_dataset, data_collator


train_dataset, test_dataset, data_collator = load_dataset(
    train_path, test_path, tokenizer)
print('dataset loaded')

# load model
model = AutoModelWithLMHead.from_pretrained(
    "gpt2", cache_dir=cache_dir)
print('model loaded')

model.resize_token_embeddings(len(tokenizer))
print('model resized')

training_args = TrainingArguments(
    output_dir=output_path,  # The output directory
    overwrite_output_dir=True,  # overwrite the content of the output directory
    num_train_epochs=3,  # number of training epochs
    per_device_train_batch_size=16,  # batch size for training
    per_device_eval_batch_size=16,  # batch size for evaluation
    eval_steps=400,  # Number of update steps between two evaluations.
    save_steps=800,  # after # steps model is saved
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    dataloader_drop_last=True,  # drops data that doesn't fit the batch size
    # gradient_accumulation_steps=16,  # saves gpu memory
)
print('used device: ' + str(training_args.device))
print('used gpus: ' + str(training_args.n_gpu))

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    prediction_loss_only=True,
)
print('trainer setup - start training')

trainer.train()
trainer.save_model()
tokenizer.save_pretrained(output_path)
print('DONE!')
