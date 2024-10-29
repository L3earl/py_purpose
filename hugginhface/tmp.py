import pandas as pd
from transformers import pipeline

# 엑셀 파일 불러오기 (예시: "models.xlsx"로 가정)
df = pd.read_excel('models.xlsx')

# 특정 모델 이름에 해당하는 파라미터 불러오기
def get_model_parameters(model_name):
    # 모델 이름에 해당하는 행을 찾고 첫 번째 행 반환
    row = df[df['model_name'] == model_name].iloc[0]

    # 파라미터가 있을 때만 값을 설정하고, 없으면 None을 유지하여 기본값을 사용
    params = {
        'max_tokens': row['max_tokens'] if pd.notna(row['max_tokens']) else None,
        'temperature': row['temperature'] if pd.notna(row['temperature']) else None,
        'top_p': row['top-p'] if pd.notna(row['top-p']) else None,
        'top_k': row['top-k'] if pd.notna(row['top-k']) else None,
        'repetition_penalty': row['repetition_penalty'] if pd.notna(row['repetition_penalty']) else None
    }

    # None 값을 제거하여 기본값을 사용하게 함
    params = {k: v for k, v in params.items() if v is not None}
    
    return params

# 모델 결과를 출력하는 함수
def get_result(model_name, prompt):
    # 모델 파라미터 불러오기
    params = get_model_parameters(model_name)

    # 모델 불러오기 (huggingface pipeline)
    model = pipeline('text-generation', model=model_name, device=0)  # 'device=0'은 GPU 사용

    # 파라미터 적용해서 결과 생성 (파라미터가 있을 경우만 전달)
    result = model(prompt, **params)
    
    return result[0]['generated_text']

# 예시로 특정 모델과 프롬프트 사용
model_name = "meta-llama/Llama-3.1-8B-Instruct"
prompt = "What is the future of AI?"

# 결과 호출
generated_text = get_result(model_name, prompt)
print(generated_text)
