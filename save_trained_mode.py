import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

def save_model(model_name, model_path, tokenizer_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    model_pegasus.save_pretrained(model_path)
    tokenizer.save_pretrained(tokenizer_path)

def get_prediction(model_path, tokenizer_path, input):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    pipe = pipeline("summarization", model=model_pegasus,tokenizer=tokenizer)

    gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}
    return pipe(dialogue, **gen_kwargs)[0]["summary_text"]

if __name__=='__main__':
    model_name = 'AbhayaHanuma/pegasus-samsum-model'

    model_path = 'artifacts/model_trainer/pegasus-samsum-model'
    tokenizer_path = 'artifacts/model_trainer/tokenizer'

    choice = int(input('Enter 1 For saving mode, 2 for running prediction: '))

    if choice==1:
        save_model(model_name, model_path, tokenizer_path)

    elif choice==2:
        dialogue = '''
              Matt: Do you want to go for date? Agnes: Wow! You caught me out with this question Matt. Matt: Why? Agnes: I simply didn't expect this from you. Matt: Well, expect the unexpected. Agnes: Can I think about it? Matt: What is there to think about? Agnes: Well, I don't really know you. Matt: This is the perfect time to get to know eachother Agnes: Well that's true. Matt: So let's go to the Georgian restaurant in Kazimierz. Agnes: Now your convincing me. Matt: Cool, saturday at 6pm? Agnes: That's fine. Matt: I can pick you up on the way to the restaurant. Agnes: That's really kind of you. Matt: No problem. Agnes: See you on saturday. Matt: Yes, looking forward to it. Agnes: Me too.
           '''
        summary = get_prediction(model_path, tokenizer_path, dialogue)
        print(f'Dialogue - \n{dialogue}\n\n')
        print(f'Summary - \n{summary}')