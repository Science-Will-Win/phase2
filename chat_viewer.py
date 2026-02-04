"""
Chat Log Viewer - Script to display JSON chat logs nicely in browser

Usage:
    python chat_viewer.py
    python chat_viewer.py --local
    uv run chat_viewer.py
"""

import argparse
import json
import os
import webbrowser
import tempfile
import sys
from pick import pick
from utils.paths import set_local_mode, get_log_dir

# Will be set dynamically based on --local flag
DEFAULT_LOG_DIR = None


def get_log_files(log_dir):
    """Return JSON file list sorted by newest first"""
    if not os.path.exists(log_dir):
        return []
    
    files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
    # Sort by filename newest first (timestamp is included in filename)
    files.sort(reverse=True)
    return files


def generate_html(data):
    """Convert JSON data to HTML"""
    model = data.get('model', 'Unknown')
    timestamp = data.get('timestamp', '')
    
    messages_html = ""
    for msg in data.get('conversation', []):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '').replace('<', '&lt;').replace('>', '&gt;')
        emoji = "👤" if role == "user" else "🤖"
        
        messages_html += f"""
            <div class="msg {role}">
                <span class="label">{emoji} {role}</span>
                <div class="bubble">{content}</div>
            </div>"""
    
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>Chat - {model}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', 'Malgun Gothic', sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ 
            max-width: 900px; 
            margin: 0 auto; 
            background: #fff;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        .header {{
            background: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .header h1 {{ 
            margin: 0; 
            font-size: 1.4em; 
        }}
        .meta {{ 
            opacity: 0.7; 
            font-size: 0.85em; 
            margin-top: 5px; 
        }}
        .chat {{ 
            padding: 20px; 
            background: #f8f9fa; 
        }}
        .msg {{ 
            margin: 15px 0; 
        }}
        .msg.user {{ 
            text-align: right; 
        }}
        .msg.assistant {{ 
            text-align: left; 
        }}
        .label {{ 
            font-size: 0.75em; 
            color: #888; 
            display: block; 
            margin-bottom: 5px; 
        }}
        .bubble {{
            display: inline-block;
            max-width: 80%;
            padding: 15px 20px;
            border-radius: 18px;
            white-space: pre-wrap;
            word-wrap: break-word;
            text-align: left;
            line-height: 1.6;
        }}
        .user .bubble {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border-bottom-right-radius: 4px;
        }}
        .assistant .bubble {{
            background: white;
            border: 1px solid #e0e0e0;
            border-bottom-left-radius: 4px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        table {{ 
            border-collapse: collapse; 
            width: 100%; 
            margin: 10px 0; 
        }}
        th, td {{ 
            border: 1px solid #ddd; 
            padding: 8px; 
            text-align: left; 
        }}
        th {{ 
            background: #f0f0f0; 
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 {model}</h1>
            <div class="meta">{timestamp}</div>
        </div>
        <div class="chat">
            {messages_html}
        </div>
    </div>
</body>
</html>"""


def view_chat(json_path):
    """Convert JSON to HTML and open in browser (using temp file)"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    html = generate_html(data)
    
    # Create temp HTML file and open in browser
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', 
                                      encoding='utf-8', delete=False) as f:
        f.write(html)
        temp_path = f.name
    
    print(f"\nOpening in browser: {os.path.basename(json_path)}")
    webbrowser.open(f'file://{temp_path}')


def main():
    global DEFAULT_LOG_DIR
    
    parser = argparse.ArgumentParser(description="Chat Log Viewer")
    parser.add_argument("--local", action="store_true",
                        help="Use local paths instead of server paths (default: server)")
    args = parser.parse_args()
    
    # Set environment based on --local flag
    set_local_mode(args.local)
    DEFAULT_LOG_DIR = get_log_dir()
    
    # Use configured log directory
    log_dir = DEFAULT_LOG_DIR
    if not os.path.isabs(log_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(script_dir, log_dir)
    
    # Get JSON file list
    files = get_log_files(log_dir)
    
    if not files:
        print(f"No JSON files in {DEFAULT_LOG_DIR} folder.")
        sys.exit(1)
    
    while True:
        # Provide arrow key selection UI with pick
        # quit_keys: pressing 'q' or 'Q' returns (None, None)
        title = f"Chat Logs ({DEFAULT_LOG_DIR})\n[Up/Down: Navigate] [Enter: Select] [q: Quit]"
        selected, index = pick(files, title, indicator=">",
                               quit_keys=[ord('q'), ord('Q')])
        
        # Exit with 'q' key (quit_keys returns None)
        if selected is None:
            print("\nExiting.")
            break
        
        # Open selected file
        json_path = os.path.join(log_dir, selected)
        view_chat(json_path)


if __name__ == "__main__":
    main()
