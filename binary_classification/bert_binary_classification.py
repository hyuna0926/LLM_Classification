import os
import random
import time
import datetime
import torch
import argparse
import re
import pandas as pd
import numpy as np

from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report


#### 데이터 로드 및 전처리
def load_data(args):
    train_df = pd.read_csv(args.train_data)
    val_df = pd.read_csv(args.val_data)

    train_df.fillna('', inplace=True)
    val_df.fillna('', inplace=True)

    train_df = train_df
    val_df = val_df
    
    # 간단한 정제 함수
    def clean_text(text):
        text = re.sub(r'\[.*?\]', '', text)  # 대괄호 안 텍스트 제거
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # URL 제거
        text = re.sub(r'<.*?>+', '', text)  # HTML 태그 제거
        text = re.sub(r'[^가-힣\s]', '', text)  # 한글, 공백 제외한 문자 제거
        return text

    train_df['cleaned_body'] = train_df['body'].apply(clean_text)
    train_df['combined_text'] = train_df['title'] + ' ' + train_df['subtitle'] + ' ' + train_df['cleaned_body'] + ' ' + train_df['caption']

    train_document = train_df.combined_text.tolist()
    train_labels = train_df.label.tolist()
    
    val_df['cleaned_body'] = val_df['body'].apply(clean_text)
    val_df['combined_text'] = val_df['title'] + ' ' + val_df['subtitle'] + ' ' + val_df['cleaned_body'] + ' ' + val_df['caption']

    val_document = val_df.combined_text.tolist()
    val_labels = val_df.label.tolist()
    
    return train_document, train_labels, val_document, val_labels


#### 토큰화
def add_special_token(document):
    # 문장 앞 뒤에 추가해줘야 함
    added = ["[CLS]" + str(sentence) + "[SEP]" for sentence in document]
    return added

def tokenization(document, mode="huggingface"):
    # 토큰화 진행
    if mode == "huggingface":
        tokenizer = BertTokenizer.from_pretrained(
                'bert-base-multilingual-cased', 
                do_lower_case=False,
                )
        tokenized = [tokenizer.tokenize(sentence) for sentence in document]
        ids = [tokenizer.convert_tokens_to_ids(sentence) for sentence in tokenized]
        return ids

def padding(ids, args):
    # 패딩(빈 공간을 채우는 작업)
    ids = pad_sequences(ids, maxlen=args.max_len, dtype="long", truncating='post', padding='post')
    return ids

def attention_mask(ids):
    # 입력으로 받은 토큰들이 실제로 중요한 정보인지, 아니면 패딩된 토큰인지를 구분하기 위한 어텐션 마스크를 생성
    masks = []
    for id in ids:
        mask = [float(i>0) for i in id]
        masks.append(mask)
    return masks

### 토큰화 진행
def train_val_data_process(args):
    train_document, train_labels, val_document, val_labels = load_data(args)
    train_document = add_special_token(train_document)
    train_ids = tokenization(train_document)
    train_ids = padding(train_ids, args)
    train_masks = attention_mask(train_ids)
    del train_document

    val_document = add_special_token(val_document)
    val_ids = tokenization(val_document)
    val_ids = padding(val_ids, args)
    val_masks = attention_mask(val_ids)
    del val_document
    
    return train_ids, train_masks, train_labels, val_ids, val_masks, val_labels

def build_dataloader(ids, masks, label, args):
    dataloader = TensorDataset(torch.tensor(ids), torch.tensor(masks), torch.tensor(label))
    dataloader = DataLoader(dataloader, sampler=RandomSampler(dataloader), batch_size=args.batch_size)
    return dataloader

def build_model(args):
    model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=args.num_labels)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"{torch.cuda.get_device_name(0)} available")
        model = model.cuda()
    else:
        device = torch.device("cpu")
        print("no GPU available")
        model = model
    return model, device

### 테스트 결과
def test(test_dataloader, model, device):
    model.eval()
    total_accuracy = 0
    all_preds = []
    all_labels = []
    for batch in test_dataloader:
        batch = tuple(index.to(device) for index in batch)
        ids, masks, labels = batch
        with torch.no_grad():
            outputs = model(ids, token_type_ids=None, attention_mask=masks)
        # 예측값 추출
        preds = torch.argmax(outputs.logits, dim=1).cpu().detach().numpy()
        true_labels = labels.cpu().numpy()
        
        # 정확도 계산을 위해 리스트에 저장
        all_preds.extend(preds)
        all_labels.extend(true_labels)

        # 배치별 정확도 계산
        accuracy = accuracy_score(true_labels, preds)
        total_accuracy += accuracy
        
    avg_accuracy = total_accuracy/len(test_dataloader)     
    report = classification_report(all_labels, all_preds, target_names=['Real News', 'Fake News'])
    
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    print(report)
    print(time.strftime('%Y.%m.%d - %H:%M:%S'))
    
    filename='0925_evaluation_results.txt'
    # 결과를 텍스트 파일로 저장
    with open(filename, 'w') as f:
        f.write(f'Validation Accuracy: {accuracy * 100:.2f}%\n\n')
        f.write(report)
        
    return avg_accuracy

### 훈련
def train(train_dataloader, test_dataloader, args):
    model, device = build_model(args)
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)*args.epochs)
    random.seed(args.seed_val)
    np.random.seed(args.seed_val)
    torch.manual_seed(args.seed_val)
    torch.cuda.manual_seed_all(args.seed_val)
    model.zero_grad()
    
    for epoch in range(0, args.epochs):
        model.train()
        total_loss, total_accuracy = 0, 0
        print("-"*30)
        for step, batch in enumerate(train_dataloader):
            if step % 500 == 0 :
                 print(f"Epoch : {epoch+1} in {args.epochs} / Step : {step}", time.strftime('%Y.%m.%d - %H:%M:%S'))

            batch = tuple(index.to(device) for index in batch)
            ids, masks, labels, = batch

            outputs = model(ids, token_type_ids=None, attention_mask=masks, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            pred = [torch.argmax(logit).cpu().detach().item() for logit in outputs.logits]
            true = [label for label in labels.cpu().numpy()]
            accuracy = accuracy_score(true, pred)
            total_accuracy += accuracy

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            
        avg_loss = total_loss / len(train_dataloader)
        avg_accuracy = total_accuracy/len(train_dataloader)
        print(f" {epoch+1} Epoch Average train loss :  {avg_loss}", time.strftime('%Y.%m.%d - %H:%M:%S'))
        print(f" {epoch+1} Epoch Average train accuracy :  {avg_accuracy}", time.strftime('%Y.%m.%d - %H:%M:%S'))

        acc = test(test_dataloader, model, device)
        os.makedirs("results", exist_ok=True)
        f = os.path.join("results", f'epoch_{epoch+1}_evalAcc_{acc*100:.0f}.pth')
        torch.save(model.state_dict(), f)
        print('Saved checkpoint:', f)
        
    # 모델 저장
    output_dir = './0925_bert_saved_model'  # 모델을 저장할 경로 지정
    model.cpu()  # 모델을 저장하기 전에 CPU로 이동
    model.save_pretrained(output_dir)


def run(args):
    train_ids, train_masks, train_labels, test_ids, test_masks, test_labels = train_val_data_process(args)
    print('data loader start', time.strftime('%Y.%m.%d - %H:%M:%S'))
    train_dataloader = build_dataloader(train_ids, train_masks, train_labels, args)
    test_dataloader = build_dataloader(test_ids, test_masks, test_labels, args)
    print('data loader end', time.strftime('%Y.%m.%d - %H:%M:%S'))
    train(train_dataloader, test_dataloader, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-train_data", default="/mnt/d/LLM/data/[Data 3] Kor - Fake News Detection/train.csv")
    parser.add_argument("-val_data", default="/mnt/d/LLM/data/[Data 3] Kor - Fake News Detection/valid.csv")

    parser.add_argument("-max_len", default=128, type=int)
    parser.add_argument("-batch_size", default=32, type=int)
    parser.add_argument("-num_labels", default=2, type=int)
    parser.add_argument("-epochs", default=4, type=int)
    parser.add_argument("-seed_val", default=42, type=int)

    args = parser.parse_args()
    run(args)