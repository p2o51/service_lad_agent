import os
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
from langchain.prompts import PromptTemplate
from langchain_core.runnables import chain
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from dotenv import load_dotenv
from google import genai
from google.genai import types
from types import SimpleNamespace


load_dotenv()
google_api_key = os.environ.get('GOOGLE_API_KEY')

if not google_api_key:
    print("没有找到 Google API 密钥，请确保在 .env 文件中设置 GOOGLE_API_KEY")
    exit()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=google_api_key,
    temperature=0.3
)

SearchUrl = 'https://c.y.qq.com/soso/fcgi-bin/client_search_cp'
LyricUrl = 'https://c.y.qq.com/lyric/fcgi-bin/fcg_query_lyric_new.fcg'
headers = {
    'referer': 'https://y.qq.com/',
    'user-agent': 'Mozilla/5.0'
}
def search_song(keyword):
    response = requests.get(
        SearchUrl, 
        headers=headers, 
        params={
            'w': keyword,
            'p': 1,
            'n': 3,
            'format': 'json'
        }
    )
    return response.json()

def dict_to_obj(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_obj(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_obj(v) for v in d]
    else:
        return d

def get_lyric(songmid):
    """获取歌词"""
    params = {
        'songmid': songmid,
        'format': 'json',
        'nobase64': '1',
    }
    response = requests.get(LyricUrl, headers=headers, params=params)
    return response.json()

def save_lrc(lyric_content, filename):
    """
    保存歌词为lrc文件
    :param lyric_content: 歌词内容
    :param filename: 文件名（不需要包含.lrc后缀）
    """
    # 确保文件名以.lrc结尾
    if not filename.endswith('.lrc'):
        filename += '.lrc'
    
    # 创建lyrics文件夹（如果不存在）
    if not os.path.exists('lyrics'):
        os.makedirs('lyrics')
    
    # 完整的文件路径
    filepath = os.path.join('lyrics', filename)
    
    # 写入文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(lyric_content)
    
    return filepath

# 使用示例
def download_song_lyric(keyword):
    """
    下载歌曲歌词的完整流程
    :param keyword: 搜索关键词
    """
    # 1. 搜索歌曲
    result = search_song(keyword)
    result_obj = dict_to_obj(result)
    
    # 2. 获取第一首歌的songmid
    songmid = result_obj.data.song.list[0].songmid
    song_name = result_obj.data.song.list[0].songname
    singer_name = result_obj.data.song.list[0].singer[0].name
    
    # 3. 获取歌词
    lyric_json = get_lyric(songmid)
    
    # 4. 保存歌词
    if lyric_json['lyric']:
        filename = f"{song_name}-{singer_name}"
        filepath = save_lrc(lyric_json['lyric'], filename)
        print(f"歌词已保存到: {filepath}")
    else:
        print("未找到歌词")

def search_lyric_meaning(context: str, text: str) -> str:

    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.0-flash"
    prompt = f"""
    请分析以下歌词的隐喻，引用，背后故事，并结合歌曲的背景信息。尽可能简洁，当你引用信息的时候请标明来源。请你直接返回分析结果。
    背景信息：（{context}）。
    歌词：{text}
    """

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    tools = [
        types.Tool(google_search=types.GoogleSearch())
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        tools=tools,
        response_mime_type="text/plain",
    )
    
    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        return ("\n\n歌词：" + text + "\n\n分析：" + response.text)
    
    except Exception as e:
        return f"分析出错：{str(e)}"

# ... existing code ...

def chunk_lyrics(lyrics, chunk_size=5, overlap=1):
    """
    将歌词分成指定大小的块，相邻块之间有重叠
    
    参数:
        lyrics (str): 歌词文本，可以包含换行符
        chunk_size (int): 每个块包含的行数
        overlap (int): 相邻块之间重叠的行数
        
    返回:
        list: 歌词块列表，每个元素是一个包含多行歌词的字符串
    """
    # 将歌词按行分割
    lines = lyrics.strip().split('\n')
    
    # 如果行数小于等于块大小，则直接返回整个歌词
    if len(lines) <= chunk_size:
        return [lyrics]
    
    chunks = []
    i = 0
    
    # 创建块，每次移动 (chunk_size - overlap) 行
    while i < len(lines):
        # 获取当前块的结束位置（确保不超出行数）
        end = min(i + chunk_size, len(lines))
        
        # 提取当前块的歌词行并合并成字符串
        chunk = '\n'.join(lines[i:end])
        chunks.append(chunk)
        
        # 移动到下一块的起始位置（考虑重叠）
        i += (chunk_size - overlap)
    
    return chunks

def analyze_all_chunks(context, lyrics, chunk_size=5, overlap=1):
    """
    分析所有歌词块并返回完整的分析报告
    
    参数:
        context (str): 歌曲背景信息
        lyrics (str): 完整歌词
        chunk_size (int): 每个块包含的行数
        overlap (int): 相邻块之间重叠的行数
        
    返回:
        list: 每个块的分析结果列表
    """
    # 将歌词分块
    chunks = chunk_lyrics(lyrics, chunk_size, overlap)
    
    # 存储所有块的分析结果
    all_analyses = []
    
    # 逐块分析
    for i, chunk in enumerate(chunks):
        print(f"分析第 {i+1}/{len(chunks)} 块歌词...")
        analysis = search_lyric_meaning(context, chunk)
        all_analyses.append({
            "chunk_index": i+1,
            "chunk_text": chunk,
            "analysis": analysis
        })
    
    return all_analyses

def generate_lyrics_report(context, analyses):
    prompt_template = """
    请根据以下信息生成 Pitchfork 风格的歌词分析报告，尽量结合歌词与背景进行分析，直接返回报告即可。
    信息：{context}
    分析结果：{analyses}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "lyrics", "analyses"])
    lyrics_chain = prompt | llm
    result = lyrics_chain.invoke({"context": context, "analyses": analyses})
    return result

if __name__ == '__main__':
    song_keyword = "Camila Cabello June Gloom"
    download_song_lyric(song_keyword) 
    
    context = "歌曲：June Gloom，歌手：Camila Cabello，专辑：C, XOXO"

    with open('lyrics/June Gloom (Explicit)-Camila Cabello.lrc', 'r', encoding='utf-8') as f:
        lyrics = f.read()

    analyses = analyze_all_chunks(context, lyrics)
    report = generate_lyrics_report(context, analyses)
    print(report)
    

