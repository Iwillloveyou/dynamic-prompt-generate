import os

import openai

DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "3f948467-3cc4-4d7c-8697-eed5c97f8e17")  # 替换为你的豆包API Key

# 豆包 API 配置（火山方舟地址）[citation:1]
DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"  # 以官方文档为准
# 推荐模型 [citation:2]
MODEL_NAME = "deepseek-v3-2-251201"  # 深度思考模型
# 备选模型: "doubao-1.5-pro-32k" (通用文本生成)

# 生成参数
NUM_DESCRIPTIONS_PER_CONCEPT = 8  # 每个概念生成多少条描述
TEMPERATURE = 0.8  # 控制多样性 (0.0-1.0) [citation:3]
MAX_TOKENS = 500  # 最大输出token数

# 初始化豆包客户端（兼容OpenAI格式）[citation:1]
client = openai.OpenAI(
    api_key=DOUBAO_API_KEY,
    base_url=DOUBAO_BASE_URL
)
# 直接替换这里的 API Key
# client = openai.OpenAI(
#     api_key="sk-edcd7fc13efd4d29a54e5976bea4a75a",  # 粘贴你的密钥
#     base_url="https://api.deepseek.com/v1"
# )

try:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        # stream=true,
        messages=[
            {"role": "system", "content": "你是一个专业的自动驾驶场景描述生成器，严格按格式输出JSON列表。"},
            {"role": "user", "content": "晴天"}
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        n=1,
        stop=None
    )
    # content = response.choices[0].message.content.strip()
    # response = client.chat.completions.create(
    #     model="deepseek-chat",
    #     messages=[{"role": "user", "content": "你好"}]
    # )
    print("✅ 成功！回复：", response.choices[0].message.content)
except Exception as e:
    print("❌ 失败：", e)