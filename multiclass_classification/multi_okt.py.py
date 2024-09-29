import os
import random
import time
import datetime

import argparse
import re
import pandas as pd
import numpy as np

from konlpy.tag import Okt

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

from data_Augmentation import EDA
from multiprocessing import Pool, cpu_count

from multiprocessing import Pool, cpu_count

# Okt 객체는 멀티프로세싱 풀 외부에 정의하여 각 프로세스에서 반복 생성되지 않도록 합니다.
okt = Okt()

# 간단한 정제 함수 (멀티프로세싱에서 사용할 함수)
def clean_text(text):
    text = re.sub(r'\[.*?\]', '', text)  # 대괄호 안 텍스트 제거
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # URL 제거
    text = re.sub(r'<.*?>+', '', text)  # HTML 태그 제거
    text = re.sub(r'[^가-힣\s]', '', text)  # 한글, 공백 제외한 문자 제거
    
    # 형태소 분석 후, 명사(Noun)와 동사(Verb)만 추출하여 결합
    return ' '.join([word for word, pos in okt.pos(text) if pos in ['Noun', 'Verb']])

# 멀티프로세싱을 적용한 정제 함수
def parallel_clean_text(texts, num_workers=12):
    with Pool(processes=num_workers) as pool:  # CPU 코어 개수에 맞춰 병렬 처리
        result = pool.map(clean_text, texts)
    return result

# 레이블을 num_labels 길이의 원-핫 벡터로 변환하는 함수
def label_split(df, num_labels):
    label_binarized = np.zeros((len(df), num_labels))
    for i, labels in enumerate(df['label'].str.split(',')):
        for label in labels:
            label_binarized[i][int(label)] = 1
    return label_binarized

# 데이터 로드 및 전처리 함수
def load_data(args):
    train_df = pd.read_csv(args.train_data, sep='\t')
    val_df = pd.read_csv(args.val_data, sep='\t')   
    
    # 결측치 빈 문자열로 채우기
    train_df.fillna('', inplace=True)
    val_df.fillna('', inplace=True)
    print('null 값 채우기', time.strftime('%Y.%m.%d - %H:%M:%S'))
    
    # 텍스트 정제
    train_df['cleaned_body'] = train_df['document'].apply(clean_text)
    val_df['cleaned_body'] = val_df['document'].apply(clean_text)
    
    # 라벨을 멀티 레이블로 변환
    train_labels = label_split(train_df, args.num_labels)
    val_labels = label_split(val_df, args.num_labels)
    print('data preprocesssing', time.strftime('%Y.%m.%d - %H:%M:%S'))
    
    return train_df.cleaned_body.tolist(), train_labels, val_df.cleaned_body.tolist(), val_labels


#### 토큰화
def add_special_token(document):
    added = ["[CLS]" + str(sentence) + "[SEP]" for sentence in document]
    return added

# 토큰화 함수
def tokenize_sentence(sentence, tokenizer):
    tokenized = tokenizer.tokenize(sentence)
    token_ids = tokenizer.convert_tokens_to_ids(tokenized)
    return token_ids

# 병렬 토큰화 함수
def parallel_tokenization(documents, num_workers=12):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    with Pool(processes=num_workers) as pool:
        tokenized_ids = pool.starmap(tokenize_sentence, [(doc, tokenizer) for doc in documents])
    return tokenized_ids

def padding(ids, args):
    ids = pad_sequences(ids, maxlen=args.max_len, dtype="long", truncating='post', padding='post')
    return ids

def attention_mask(ids):
    masks = [[float(i > 0) for i in seq] for seq in ids]
    return masks

### 토큰화 및 전처리
def train_val_data_process(args):
    train_document, train_labels, val_document, val_labels = load_data(args)
    print('data preprocessing', time.strftime('%Y.%m.%d - %H:%M:%S'))
    
    # 스페셜 토큰 추가
    train_document = add_special_token(train_document)
    val_document = add_special_token(val_document)
    print('add_special_token', time.strftime('%Y.%m.%d - %H:%M:%S'))
    
    # 멀티프로세싱으로 토큰화
    train_ids = parallel_tokenization(train_document, num_workers=12)
    val_ids = parallel_tokenization(val_document, num_workers=12)
    print('parallel_tokenization', time.strftime('%Y.%m.%d - %H:%M:%S'))
    
    # 패딩 및 마스크 생성
    train_ids = padding(train_ids, args)
    train_masks = attention_mask(train_ids)
    del train_document

    val_ids = padding(val_ids, args)
    val_masks = attention_mask(val_ids)
    del val_document
    print('padding, mask', time.strftime('%Y.%m.%d - %H:%M:%S'))
    return train_ids, train_masks, train_labels, val_ids, val_masks, val_labels

### 데이터 로더 구성
def build_dataloader(ids, masks, label, args):
    dataloader = TensorDataset(torch.tensor(ids), torch.tensor(masks), torch.tensor(label))
    dataloader = DataLoader(dataloader, sampler=RandomSampler(dataloader), batch_size=args.batch_size, num_workers=10)
    print('data loader', time.strftime('%Y.%m.%d - %H:%M:%S'))
    return dataloader

### BERT 모델 빌드 (멀티레이블 분류 지원)
class BertForMultiLabelClassification(nn.Module):
    def __init__(self, num_labels):
        super(BertForMultiLabelClassification, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=num_labels)
        self.sigmoid = nn.Sigmoid()  # 각 라벨의 확률을 계산

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = self.sigmoid(logits)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()  # 멀티레이블 분류를 위한 손실 함수
            loss = loss_fct(logits, labels.float())

        return loss, probs

def build_model(args):
    model = BertForMultiLabelClassification(num_labels=args.num_labels)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"{torch.cuda.get_device_name(0)} available")
        model = model.cuda()
    else:
        device = torch.device("cpu")
        print("no GPU available")
        model = model
    return model, device

### 테스트 함수 (멀티레이블 분류)
def test(test_dataloader, model, device):
    model.eval()
    total_accuracy = 0
    all_preds = []
    all_labels = []
    class_names = ['Origin Discrimination', 'Appearance Discrimination', 'Political Orientation Discrimination', 
                   'Hate Speech', 'Age Discrimination', 'Gender Discrimination', 'Racial Discrimination', 
                   'Religious Discrimination', 'Not Hate Speech']
    
    for batch in test_dataloader:
        batch = tuple(index.to(device) for index in batch)
        ids, masks, labels = batch
        with torch.no_grad():
            _, probs = model(ids, attention_mask=masks)

        # 라벨별로 0.5 이상인 경우 1로 간주하여 예측
        preds = (probs > 0.5).int().cpu().numpy()
        true_labels = labels.cpu().numpy()
    
        all_preds.extend(preds)
        all_labels.extend(true_labels)
        
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 마이크로, 매크로 지표 계산
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

    # 정확도
    avg_accuracy = accuracy_score(all_labels, all_preds)

    # 결과 출력
    print(f"Valid AVG accuracy: {avg_accuracy:.2f}\n")
    print(f"Micro-average Precision: {precision_micro:.2f}, Recall: {recall_micro:.2f}, F1: {f1_micro:.2f}\n")
    print(f"Macro-average Precision: {precision_macro:.2f}, Recall: {recall_macro:.2f}, F1: {f1_macro:.2f}\n")
    
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("Classification Report:\n", report)

    filename = '0928_results_evaluation.txt'
    with open(filename, 'w') as f:
        f.write(f"Micro-average Precision: {precision_micro:.2f}, Recall: {recall_micro:.2f}, F1: {f1_micro:.2f}")
        f.write(f"Macro-average Precision: {precision_macro:.2f}, Recall: {recall_macro:.2f}, F1: {f1_macro:.2f}")
        f.write(f'Validation Accuracy: {avg_accuracy * 100:.2f}%\n\n')
        f.write(report)
        f.write(time.strftime('%Y.%m.%d - %H:%M:%S'))
        
    return avg_accuracy

### 훈련 함수 (멀티레이블 분류)
def train(train_dataloader, test_dataloader, args):
    model, device = build_model(args)
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * args.epochs)
    
    random.seed(args.seed_val)
    np.random.seed(args.seed_val)
    torch.manual_seed(args.seed_val)
    torch.cuda.manual_seed_all(args.seed_val)
    
    model.zero_grad()
    
    for epoch in range(0, args.epochs):
        model.train()
        total_loss, total_accuracy = 0, 0
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        for step, batch in enumerate(train_dataloader):
            if step % 100 == 0:
                print(f"Epoch : {epoch+1}/{args.epochs}, Step: {step}", time.strftime('%Y.%m.%d - %H:%M:%S'))
            # print('step', step, time.strftime('%Y.%m.%d - %H:%M:%S'))
            batch = tuple(index.to(device) for index in batch)
            # print('batch', batch)
            ids, masks, labels = batch

            outputs = model(ids, attention_mask=masks, labels=labels)
            loss = outputs[0]
            # print('loss', loss)
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_loss:.2f}")
        
        acc = test(test_dataloader, model, device)
        print(f'Validation Accuracy: {acc:.2f}')
        
        torch.save(model.state_dict(), f'0929_model_dict/checkpoint_epoch_{epoch+1}.pth')
        print(time.strftime('%Y.%m.%d - %H:%M:%S'))

    # 모델 저장
    # output_dir = './0928_data1_bert_saved_model'  # 모델을 저장할 경로 지정
    # model.cpu()  # 모델을 저장하기 전에 CPU로 이동
    # model.save_pretrained(output_dir)    
    
### 메인 실행 함수
def run(args):
    train_ids, train_masks, train_labels, test_ids, test_masks, test_labels = train_val_data_process(args)
    train_dataloader = build_dataloader(train_ids, train_masks, train_labels, args)
    test_dataloader = build_dataloader(test_ids, test_masks, test_labels, args)
    train(train_dataloader, test_dataloader, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-train_data", default="/mnt/d/LLM/data/[Data 1] Kor - Hate Speech Detection/train.txt")
    parser.add_argument("-val_data", default="/mnt/d/LLM/data/[Data 1] Kor - Hate Speech Detection/valid.txt")

    
    parser.add_argument("-max_len", default=128, type=int)
    parser.add_argument("-batch_size", default=32, type=int)
    parser.add_argument("-num_labels", default=9, type=int)  # 9개의 라벨
    parser.add_argument("-epochs", default=4, type=int)
    parser.add_argument("-seed_val", default=42, type=int)

    args = parser.parse_args()
    run(args)
