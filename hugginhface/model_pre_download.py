from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv
import os
from transformers import pipeline
import torch

# CUDA가 사용 가능한지 확인하여 device 설정
device = 0 if torch.cuda.is_available() else -1
# 출력해서 현재 선택된 device 확인
print(f"Using device: {device}")

# .env 파일에서 환경 변수 로드
load_dotenv()
# .env 파일에서 Hugging Face 토큰 불러오기
token = os.getenv('HUGGINGFACE_TOKEN')

# 허깅페이스의 모델들을 불러와서 프롬프트를 입력하고, 결과를 출력하는 함수
def get_result(model_name, prompt):
    # 모델 불러오기
    model = pipeline('text-generation', model=model_name, token=token, device=device)
    
    # 결과 출력
    # max_length: 최대 길이, max_new_tokens: 최대 토큰 수
    # result = model(prompt, max_length=1024, max_new_tokens=1024)
    result = model(prompt, max_length=512)
    
    return result[0]['generated_text']

# 사용할 모델 목록
## 기본적으로 Meta-Llama 모델을 사용 (멘토님 조언)
### 241004 기준 3.2버전이 최신임
### intruct 모델은 사용자의 지시문(instruction)을 더 잘 이해하고, 이에 맞는 적절한 답변을 생성할 수 있도록 미세 조정(fine-tuning)된 모델 - 사용
### Evals 모델은 모델의 성능을 평가하고 검증하기 위한 특화된 모델
### Llama-Guard 모델은 일반적인 Llama 모델과 비교하여 입력 및 출력의 안전성(위험한 콘텐츠 감지, 방지) 을 보장하기 위한 특화된 모델
model_lst = [
            # 'meta-llama/Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.2-3B-Instruct',
            #  'meta-llama/Llama-3.1-8B-Instruct', 
            #  'meta-llama/Llama-3.1-70B-Instruct',             
            #  "mistralai/Mixtral-8x7B-Instruct-v0.1",
            #  "mistralai/Mixtral-8x22B-Instruct-v0.1",
            # "Qwen/Qwen2.5-1.5B-Instruct", 
            # "Qwen/Qwen2.5-3B-Instruct", 
            #  "Qwen/Qwen2.5-7B-Instruct", 
            #  "Qwen/Qwen2.5-14B-Instruct", 
             "Qwen/Qwen2.5-32B-Instruct"
            #  "Qwen/Qwen2.5-72B-Instruct",
            #  'meta-llama/Llama-3.1-405B-Instruct',
            # "upstage/solar-pro-preview-instruct"
             ]

# 사용할 모델 미리 다운 받아놓기 (허깅페이스에서 모델 다운받는데 오래 걸림)
for model_name in model_lst:
    print(f"Downloading {model_name}...")
    print(get_result(model_name, "Hello, I'm a language model."))
    # 모델과 토크나이저 다운로드 및 로컬에 저장
    # tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    # model = AutoModel.from_pretrained(model_name, token=token)