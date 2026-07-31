"""
古诗生成系统 —— Web 界面
基于 Flask 的简易 Web 前端

启动方式: python web_app.py
访问地址: http://localhost:5000
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from flask import Flask, render_template_string, request, jsonify

from src.tone_dict import ToneDict
from src.analyzer import PoetryAnalyzer
from src.transformer_model import TransformerModel
from src.generator import PoemGenerator

app = Flask(__name__)

generator = None


def get_generator():
    global generator
    if generator is None:
        print('正在加载模型...')

        tone_dict = ToneDict()
        analyzer = PoetryAnalyzer()
        analyzer.load()

        t = TransformerModel()
        if t.load('transformer_model'):
            print('Transformer 模型已加载')
        else:
            print('警告: 未找到模型，请先在云端训练')

        generator = PoemGenerator(
            transformer_model=t,
            tone_dict=tone_dict,
            analyzer=analyzer,
        )
        print('模型加载完成!')
    return generator


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>古诗生成系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: "Microsoft YaHei", "宋体", SimSun, serif;
            background: linear-gradient(135deg, #f5f0e8 0%, #e8dcc8 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #5d4037;
            font-size: 2.2em;
            margin: 30px 0 10px;
            letter-spacing: 4px;
        }
        .subtitle {
            text-align: center;
            color: #8d6e63;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .card {
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        .card h2 {
            color: #5d4037;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #d7ccc8;
            font-size: 1.4em;
        }
        .form-group {
            margin-bottom: 18px;
        }
        label {
            display: block;
            color: #6d4c41;
            margin-bottom: 6px;
            font-weight: bold;
            font-size: 0.95em;
        }
        input, select {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #d7ccc8;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s;
            font-family: inherit;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #8d6e63;
        }
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #8d6e63, #6d4c41);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.1em;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            letter-spacing: 2px;
            font-family: inherit;
        }
        button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(109, 76, 65, 0.3);
        }
        button:active {
            transform: translateY(0);
        }
        .result {
            margin-top: 25px;
            padding: 25px;
            background: #faf7f2;
            border-radius: 8px;
            border-left: 4px solid #8d6e63;
            display: none;
        }
        .result.show { display: block; }
        .poem-text {
            font-size: 1.3em;
            line-height: 2;
            text-align: center;
            color: #3e2723;
            letter-spacing: 2px;
        }
        .poem-text .line {
            display: block;
        }
        .poem-text .line.acrostic-first {
            color: #c62828;
            font-weight: bold;
        }
        .score-info {
            margin-top: 15px;
            font-size: 0.85em;
            color: #8d6e63;
            text-align: center;
        }
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .tab {
            flex: 1;
            padding: 12px;
            text-align: center;
            background: #efebe9;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: bold;
            color: #6d4c41;
        }
        .tab.active {
            background: #8d6e63;
            color: white;
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .loading {
            text-align: center;
            padding: 20px;
            color: #8d6e63;
            display: none;
        }
        .loading.show { display: block; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📜 古诗生成系统</h1>
        <p class="subtitle">基于85万首古典诗歌训练的 Transformer 语言模型</p>

        <div class="card">
            <div class="tabs">
                <div class="tab active" onclick="switchTab('keyword')">🔑 关键词生成</div>
                <div class="tab" onclick="switchTab('acrostic')">🎭 藏头诗</div>
            </div>

            <!-- 关键词生成 Tab -->
            <div id="tab-keyword" class="tab-content active">
                <div class="form-group">
                    <label>关键词（空格分隔）</label>
                    <input type="text" id="kw-words" placeholder="例: 春风 明月 故乡" value="春风 明月">
                </div>
                <div class="form-group">
                    <label>诗体</label>
                    <select id="kw-type">
                        <option value="五言绝句">五言绝句 (4句×5字)</option>
                        <option value="七言绝句">七言绝句 (4句×7字)</option>
                        <option value="五言律诗">五言律诗 (8句×5字)</option>
                        <option value="七言律诗">七言律诗 (8句×7字)</option>
                    </select>
                </div>
                <button onclick="generateKeyword()">✨ 生成古诗</button>
            </div>

            <!-- 藏头诗 Tab -->
            <div id="tab-acrostic" class="tab-content">
                <div class="form-group">
                    <label>藏头字</label>
                    <input type="text" id="ac-head" placeholder="例: 生日快乐" value="生日快乐">
                </div>
                <div class="form-group">
                    <label>诗体（可选，默认自动选择）</label>
                    <select id="ac-type">
                        <option value="">自动选择</option>
                        <option value="五言绝句">五言绝句</option>
                        <option value="七言绝句">七言绝句</option>
                        <option value="五言律诗">五言律诗</option>
                        <option value="七言律诗">七言律诗</option>
                    </select>
                </div>
                <button onclick="generateAcrostic()">✨ 生成藏头诗</button>
            </div>

            <!-- 加载提示 -->
            <div id="loading" class="loading">⏳ 正在生成中...</div>

            <!-- 结果显示 -->
            <div id="result" class="result">
                <div id="poem-display" class="poem-text"></div>
                <div id="score-display" class="score-info"></div>
            </div>
        </div>
    </div>

    <script>
        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelector(`.tab:nth-child(${tab === 'keyword' ? 1 : 2})`).classList.add('active');
            document.getElementById(`tab-${tab}`).classList.add('active');
        }

        function showLoading() {
            document.getElementById('loading').classList.add('show');
            document.getElementById('result').classList.remove('show');
        }

        function hideLoading() {
            document.getElementById('loading').classList.remove('show');
        }

        function displayPoem(data) {
            const display = document.getElementById('poem-display');
            const scoreDisplay = document.getElementById('score-display');
            const resultDiv = document.getElementById('result');

            let html = '';
            if (data.lines) {
                data.lines.forEach(line => {
                    html += `<span class="line">${line}</span>`;
                });
            }
            display.innerHTML = html;

            let scoreHtml = `总得分: ${data.score?.toFixed(2) || '-'}`;
            if (data.tone_score !== undefined) {
                scoreHtml += ` | 平仄得分: ${(data.tone_score * 100).toFixed(0)}%`;
            }
            scoreDisplay.innerHTML = scoreHtml;
            resultDiv.classList.add('show');
        }

        function generateKeyword() {
            const words = document.getElementById('kw-words').value.trim();
            const type = document.getElementById('kw-type').value;

            if (!words) {
                alert('请输入关键词');
                return;
            }

            showLoading();
            fetch('/api/generate_keyword', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({words: words.split(/\\s+/), type: type})
            })
            .then(r => r.json())
            .then(data => {
                hideLoading();
                if (data.error) {
                    alert(data.error);
                } else {
                    displayPoem(data);
                }
            })
            .catch(err => {
                hideLoading();
                alert('生成失败: ' + err.message);
            });
        }

        function generateAcrostic() {
            const head = document.getElementById('ac-head').value.trim();
            const type = document.getElementById('ac-type').value || null;

            if (!head) {
                alert('请输入藏头字');
                return;
            }

            showLoading();
            fetch('/api/generate_acrostic', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({head: head, type: type})
            })
            .then(r => r.json())
            .then(data => {
                hideLoading();
                if (data.error) {
                    alert(data.error);
                } else {
                    displayPoem(data);
                }
            })
            .catch(err => {
                hideLoading();
                alert('生成失败: ' + err.message);
            });
        }
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/generate_keyword', methods=['POST'])
def api_keyword():
    try:
        data = request.get_json()
        words = data.get('words', [])
        poem_type = data.get('type', '五言绝句')

        if not words:
            return jsonify({'error': '关键词不能为空'})

        gen = get_generator()
        result = gen.generate_from_keywords(keywords=words, poem_type=poem_type)

        return jsonify({
            'poem': result['poem'],
            'lines': result['lines'],
            'score': result['score'],
            'tone_score': result.get('tone_score', 0),
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/generate_acrostic', methods=['POST'])
def api_acrostic():
    try:
        data = request.get_json()
        head = data.get('head', '')
        poem_type = data.get('type')

        if not head:
            return jsonify({'error': '藏头字不能为空'})

        gen = get_generator()
        result = gen.generate_acrostic(head_chars=head, poem_type=poem_type)

        return jsonify({
            'poem': result['poem'],
            'lines': result['lines'],
            'head_chars': result['head_chars'],
            'score': result['score'],
            'tone_score': result.get('tone_score', 0),
        })
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    print('=' * 50)
    print('  古诗生成系统 Web 界面')
    print('  访问: http://localhost:5000')
    print('=' * 50)
    # 预加载模型
    get_generator()
    app.run(debug=False, host='0.0.0.0', port=5000)
