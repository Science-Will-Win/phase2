"""
RLHF Preference Data Collection Tool (Streaming Version)

A lightweight HTTP server for collecting human preference data.
Features real-time streaming generation.

Usage:
    python rlhf_collect.py --data_path data/sample.json --model ministral_3_3b_instruct

Features:
    - Real-time streaming: Watch responses generate token by token
    - Automatic session detection: resumes incomplete sessions or starts new
    - Real-time saving: preferences saved immediately on selection
    - Keyboard shortcuts: 1/2 for A/B, S for skip
"""

import argparse
import atexit
import json
import os
import signal
import sys
import threading
import time
import webbrowser
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse
import mimetypes

# Early GPU Configuration (MUST be before torch import)
from utils.gpu_config import configure_gpu
configure_gpu()

import torch
import torch.nn as nn
# pick removed - using custom keyboard input for reliability

# Browser connection tracking
last_heartbeat = time.time()
browser_connected = True
HEARTBEAT_TIMEOUT = 30  # seconds (increased for long model generation)

# Threading HTTP Server for concurrent request handling
class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

# Project imports
import model as model_module
from tokenizer import TokenizerManager
from utils import get_file_config
from utils.paths import set_local_mode, get_data_dir, get_temp_data_dir, ensure_dirs
from inference import (
    calculate_uncertainty,
    sample_token,
    compute_recovery_temp,
    generate_with_refusal_streaming,
)

# Constants
PORT = 8080
# TEMP_DATA_DIR and FINAL_DATA_DIR are set dynamically based on --local flag
TEMP_DATA_DIR = None  # Will be set in main()
FINAL_DATA_DIR = None  # Will be set in main()
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preference_ui")

# Global model and tokenizer (kept in memory for streaming)
global_model = None
global_tokenizer = None
global_system_prompt = "You are a helpful AI assistant."
global_args = None
global_server = None  # HTTP server instance for cleanup


from utils.model_utils import get_model_device, get_unwrapped_model


def cleanup():
    """Clean up resources on exit."""
    global global_model, global_tokenizer, global_server
    
    print("\n[INFO] Cleaning up resources...")
    
    # Stop HTTP server if running
    if global_server is not None:
        try:
            global_server.shutdown()
            print("[INFO] HTTP server stopped")
        except Exception:
            pass
        global_server = None
    
    # Clear model from memory
    if global_model is not None:
        del global_model
        global_model = None
    
    if global_tokenizer is not None:
        del global_tokenizer
        global_tokenizer = None
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[INFO] GPU memory released")
    
    print("[INFO] Cleanup complete")


def cleanup_server():
    """Clean up server resources only (keep model in memory)."""
    global global_server
    
    if global_server is not None:
        try:
            global_server.shutdown()
            print("[INFO] HTTP server stopped")
        except Exception:
            pass
        global_server = None


def kill_nodejs_processes():
    """Kill Node.js processes to prevent GPU conflicts."""
    import subprocess
    import time
    try:
        # Terminate node.exe processes on Windows
        result = subprocess.run(
            ['taskkill', '/F', '/IM', 'node.exe'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("[INFO] Node.js processes terminated to prevent GPU conflicts")
            time.sleep(3)  # Wait for GPU resources to be released
        # If returncode != 0, node.exe doesn't exist, so ignore
    except Exception:
        pass  # Continue even if failed


def signal_handler(signum, frame):
    """Handle termination signals."""
    sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    print(f"\n[INFO] Received signal {sig_name}, shutting down...")
    cleanup()
    sys.exit(0)


class SessionManager:
    """Manages session state and persistence."""
    
    def __init__(self):
        self.session_id = None
        self.session_file = None
        self.preferences_file = None
        self.state = None
        self.current_prompt_index = 0
        self.prompts = []
        self.temperatures = []
        self.current_responses = {}  # Store generated responses for current prompt
        self.response_cache = {}  # Cache: prompt_index -> {temp_index: {"text": str, "temperature": float}}
    
    def create_new_session(self, config: dict, prompts: list, temperatures: list):
        """Create a new session."""
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        self.session_id = timestamp
        self.prompts = prompts
        self.temperatures = temperatures
        
        # Extract original data name from data_path
        data_path = config.get("data_path", "unknown")
        data_name = os.path.splitext(os.path.basename(data_path))[0]
        
        # Ensure temp directory exists (data saved here until session completes)
        os.makedirs(TEMP_DATA_DIR, exist_ok=True)
        
        # New naming format: RLHF_preference_data_{data_name}_{date}_{time}
        base_name = f"RLHF_preference_data_{data_name}_{timestamp}"
        self.session_file = os.path.join(
            TEMP_DATA_DIR, f"{base_name}_session.json"
        )
        self.preferences_file = os.path.join(
            TEMP_DATA_DIR, f"{base_name}.json"
        )
        
        self.state = {
            "session_id": self.session_id,
            "created_at": now.isoformat(),
            "config": config,
            "prompts": [{"instruction": p.get("instruction", p.get("text", str(p))), "completed": False, "chosen_idx": None, "rejected_idx": None} for p in prompts],
            "progress": {
                "total": len(prompts),
                "completed": 0
            },
            "generated_responses": {}  # prompt_index -> {temp_index: {text, temperature}}
        }
        
        self.current_prompt_index = 0
        self.current_responses = {}
        self.response_cache = {}
        self._save_state()
        
        return self.state
    
    def load_session(self, session_file: str):
        """Load an existing session."""
        with open(session_file, "r", encoding="utf-8") as f:
            self.state = json.load(f)
        
        self.session_id = self.state["session_id"]
        self.session_file = session_file
        # New naming: remove "_session" suffix and change extension
        self.preferences_file = session_file.replace("_session.json", ".json")
        self.temperatures = self.state["config"].get("temperatures", [0.3, 0.7, 1.0, 1.2])
        
        # Rebuild prompts list
        self.prompts = [{"instruction": p["instruction"]} for p in self.state["prompts"]]
        
        # Always start from first prompt
        self.current_prompt_index = 0
        
        # Restore response cache from saved state
        self.response_cache = {}
        if "generated_responses" in self.state:
            for prompt_idx, responses in self.state["generated_responses"].items():
                self.response_cache[int(prompt_idx)] = {
                    int(temp_idx): resp for temp_idx, resp in responses.items()
                }
        
        # Set current responses from cache
        self.current_responses = self.response_cache.get(self.current_prompt_index, {}).copy()
        return self.state
    
    def get_current_prompt(self):
        """Get the current prompt to generate responses for."""
        if self.current_prompt_index >= len(self.prompts):
            return None
        
        prompt_data = self.prompts[self.current_prompt_index]
        instruction = prompt_data.get("instruction", prompt_data.get("text", str(prompt_data)))
        
        # Get selection state from session
        prompt_state = self.state["prompts"][self.current_prompt_index]
        
        return {
            "prompt_index": self.current_prompt_index,
            "instruction": instruction,
            "temperatures": self.temperatures,
            "total_prompts": len(self.prompts),
            "completed": prompt_state.get("completed", False),
            "chosen_idx": prompt_state.get("chosen_idx"),
            "rejected_idx": prompt_state.get("rejected_idx")
        }
    
    def store_response(self, temp_index: int, text: str, temperature: float,
                        min_temp_used: float = None, avg_temp_used: float = None):
        """Store a generated response."""
        response_data = {
            "text": text,
            "temperature": temperature,
            "min_temp_used": min_temp_used,
            "avg_temp_used": avg_temp_used
        }
        self.current_responses[temp_index] = response_data
        
        # Also cache it in memory
        if self.current_prompt_index not in self.response_cache:
            self.response_cache[self.current_prompt_index] = {}
        self.response_cache[self.current_prompt_index][temp_index] = response_data.copy()
        
        # Persist to session file for recovery across restarts
        prompt_key = str(self.current_prompt_index)
        if prompt_key not in self.state["generated_responses"]:
            self.state["generated_responses"][prompt_key] = {}
        self.state["generated_responses"][prompt_key][str(temp_index)] = response_data.copy()
        self._save_state()
    
    def get_cached_responses(self, prompt_index: int):
        """Get cached responses for a prompt if available."""
        return self.response_cache.get(prompt_index, {})
    
    def has_cached_response(self, prompt_index: int, temp_index: int):
        """Check if a specific response is cached."""
        return prompt_index in self.response_cache and temp_index in self.response_cache[prompt_index]
    
    def get_prompt_at_index(self, index: int):
        """Get prompt at a specific index."""
        if index < 0 or index >= len(self.prompts):
            return None
        
        prompt_data = self.prompts[index]
        instruction = prompt_data.get("instruction", prompt_data.get("text", str(prompt_data)))
        
        # Get selection state from session
        prompt_state = self.state["prompts"][index] if self.state else {}
        
        return {
            "prompt_index": index,
            "instruction": instruction,
            "temperatures": self.temperatures,
            "total_prompts": len(self.prompts),
            "completed": prompt_state.get("completed", False),
            "chosen_idx": prompt_state.get("chosen_idx"),
            "rejected_idx": prompt_state.get("rejected_idx")
        }
    
    def go_to_prompt(self, index: int):
        """Move to a specific prompt index."""
        if index < 0 or index >= len(self.prompts):
            return False
        self.current_prompt_index = index
        self.current_responses = self.response_cache.get(index, {}).copy()
        return True
    
    def go_prev(self):
        """Move to the previous prompt."""
        if self.current_prompt_index > 0:
            self.current_prompt_index -= 1
            self.current_responses = self.response_cache.get(self.current_prompt_index, {}).copy()
            return True
        return False
    
    def go_next(self):
        """Move to the next prompt (for navigation without saving)."""
        if self.current_prompt_index < len(self.prompts) - 1:
            self.current_prompt_index += 1
            self.current_responses = self.response_cache.get(self.current_prompt_index, {}).copy()
            return True
        return False
    
    def save_preference(self, chosen_idx: int, rejected_idx: int):
        """Save a preference choice."""
        if self.current_prompt_index >= len(self.prompts):
            return False
        
        prompt_data = self.prompts[self.current_prompt_index]
        instruction = prompt_data.get("instruction", prompt_data.get("text", str(prompt_data)))
        
        chosen_resp = self.current_responses.get(chosen_idx)
        rejected_resp = self.current_responses.get(rejected_idx)
        
        if not chosen_resp or not rejected_resp:
            return False
        
        # Check if this prompt was already completed
        already_completed = self.state["prompts"][self.current_prompt_index].get("completed", False)
        
        # Save to preferences file (JSON array)
        preference = {
            "prompt": instruction,
            "chosen": chosen_resp["text"],
            "rejected": rejected_resp["text"],
            "chosen_temp": chosen_resp["temperature"],
            "rejected_temp": rejected_resp["temperature"],
            "prompt_index": self.current_prompt_index,
            "chosen_idx": chosen_idx,
            "rejected_idx": rejected_idx,
            "timestamp": datetime.now().isoformat()
        }
        
        # Load existing preferences or create empty list
        preferences = []
        if os.path.exists(self.preferences_file):
            try:
                with open(self.preferences_file, "r", encoding="utf-8") as f:
                    preferences = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                preferences = []
        
        # Update or append preference (replace if same prompt_index exists)
        updated = False
        for i, p in enumerate(preferences):
            if p.get("prompt_index") == self.current_prompt_index:
                preferences[i] = preference
                updated = True
                break
        if not updated:
            preferences.append(preference)
        
        # Save as JSON array
        with open(self.preferences_file, "w", encoding="utf-8") as f:
            json.dump(preferences, f, ensure_ascii=False, indent=2)
        
        # Save selection info to state
        self.state["prompts"][self.current_prompt_index]["chosen_idx"] = chosen_idx
        self.state["prompts"][self.current_prompt_index]["rejected_idx"] = rejected_idx
        self.state["prompts"][self.current_prompt_index]["completed"] = True
        
        # Only increment count if this is the first time completing this prompt
        if not already_completed:
            self.state["progress"]["completed"] += 1
        
        self._save_state()
        
        # Check if all prompts are completed -> move to final location
        if self.state["progress"]["completed"] == self.state["progress"]["total"]:
            self._move_to_final()
        
        return True
    
    def skip_current(self):
        """Skip the current prompt."""
        if self.current_prompt_index >= len(self.prompts):
            return False
        
        self.current_prompt_index += 1
        self.current_responses = {}
        
        return True
    
    def clear_preference(self, prompt_index: int = None):
        """Clear preference for a specific prompt."""
        if prompt_index is None:
            prompt_index = self.current_prompt_index
        
        if prompt_index < 0 or prompt_index >= len(self.prompts):
            return False
        
        prompt_state = self.state["prompts"][prompt_index]
        if prompt_state.get("completed", False):
            prompt_state["completed"] = False
            prompt_state["chosen_idx"] = None
            prompt_state["rejected_idx"] = None
            self.state["progress"]["completed"] -= 1
            self._save_state()
        return True
    
    def clear_all_preferences(self):
        """Clear all preferences."""
        for prompt_state in self.state["prompts"]:
            prompt_state["completed"] = False
            prompt_state["chosen_idx"] = None
            prompt_state["rejected_idx"] = None
        self.state["progress"]["completed"] = 0
        
        # Clear preferences file (save empty array)
        with open(self.preferences_file, "w", encoding="utf-8") as f:
            json.dump([], f)
        
        self._save_state()
        return True
    
    def delete_session(self):
        """Delete current session files from temp_data."""
        deleted = []
        if self.session_file and os.path.exists(self.session_file):
            os.remove(self.session_file)
            deleted.append(self.session_file)
        if self.preferences_file and os.path.exists(self.preferences_file):
            os.remove(self.preferences_file)
            deleted.append(self.preferences_file)
        return deleted
    
    def _save_state(self):
        """Save current state to file."""
        with open(self.session_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
    
    def _move_to_final(self):
        """Copy completed session to data with chosen responses only."""
        os.makedirs(FINAL_DATA_DIR, exist_ok=True)
        
        final_pref = os.path.join(FINAL_DATA_DIR, os.path.basename(self.preferences_file))
        
        # Extract chosen responses only and save to data/
        if os.path.exists(self.preferences_file):
            with open(self.preferences_file, "r", encoding="utf-8") as f:
                preferences = json.load(f)
            
            # Save only prompt and chosen as JSON array
            chosen_only_list = []
            for data in preferences:
                chosen_only = {
                    "prompt": data["prompt"],
                    "chosen": data["chosen"]
                }
                chosen_only_list.append(chosen_only)
            
            with open(final_pref, "w", encoding="utf-8") as f:
                json.dump(chosen_only_list, f, ensure_ascii=False, indent=2)
            
            print(f"[INFO] Session completed! Data saved to: {final_pref}")
        
        # Keep temp_data files (do not delete)
        print(f"[INFO] temp_data preserved: {self.preferences_file}")
    
    def get_status(self):
        """Get current progress status."""
        return {
            "session_id": self.session_id,
            "total": len(self.prompts),
            "completed": self.state["progress"]["completed"] if self.state else 0,
            "remaining": len(self.prompts) - (self.state["progress"]["completed"] if self.state else 0),
            "current_index": self.current_prompt_index,
            "preferences_file": self.preferences_file,
            "session_file": self.session_file
        }


# Global session manager
session_manager = SessionManager()


def generate_streaming(prompt: str, temperature: float, max_length: int = 32768):
    """
    Generate response token by token, yielding each token as it's generated.
    Wrapper around inference.py's generate_with_refusal_streaming.
    
    Uses the shared generator from inference.py for code reuse.
    """
    global global_model, global_tokenizer, global_system_prompt, global_args
    
    if global_model is None or global_tokenizer is None:
        yield {"error": "Model not loaded"}
        return
    
    # Build messages
    messages = [
        {"role": "system", "content": global_system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    # Tokenize
    device = get_model_device(global_model)
    try:
        inputs = global_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        if isinstance(inputs, torch.Tensor):
            input_ids = inputs.to(device)
            attention_mask = torch.ones_like(input_ids)
        else:
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)
    except Exception as e:
        text = f"{global_system_prompt}\n\nUser: {prompt}\nAssistant:"
        inputs = global_tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = torch.ones_like(input_ids)
    
    # Build args object for generate_with_refusal_streaming
    class GenArgs:
        pass
    
    args = GenArgs()
    args.temperature = temperature
    args.max_length = max_length
    args.top_k = getattr(global_args, 'top_k', 50) if global_args else 50
    args.refusal_threshold = getattr(global_args, 'refusal_threshold', 3.0) if global_args else 3.0
    # Disable refusal if --no_refusal is set
    if getattr(global_args, 'no_refusal', False):
        args.refusal_threshold = float('inf')
    args.refusal_max_retries = getattr(global_args, 'refusal_max_retries', 3) if global_args else 3
    args.refusal_temp_decay = getattr(global_args, 'refusal_temp_decay', 0.8) if global_args else 0.8
    args.refusal_min_temp = getattr(global_args, 'refusal_min_temp', 0.4) if global_args else 0.4
    args.refusal_recovery_tokens = getattr(global_args, 'refusal_recovery_tokens', 3) if global_args else 3
    args.refusal_recovery_method = getattr(global_args, 'refusal_recovery_method', 'exponential') if global_args else 'exponential'
    args.random_seed = -1  # Always use random seed for RLHF diversity
    args.debug = getattr(global_args, 'debug', False) if global_args else False
    
    # Package inputs for the generator
    inputs_dict = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    
    # Use inference.py's generator
    for chunk in generate_with_refusal_streaming(global_model, global_tokenizer, inputs_dict, args):
        yield chunk


class PreferenceHandler(BaseHTTPRequestHandler):
    """HTTP request handler for preference collection with SSE support."""
    
    def log_message(self, format, *args):
        """Suppress default logging for cleaner output."""
        pass
    
    def send_json(self, data, status=200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
    
    def serve_static_file(self, filepath):
        """Serve a static file."""
        if not os.path.exists(filepath):
            self.send_error(404)
            return
        
        mime_type, _ = mimetypes.guess_type(filepath)
        if mime_type is None:
            mime_type = "application/octet-stream"
        
        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.end_headers()
        
        with open(filepath, "rb") as f:
            self.wfile.write(f.read())
    
    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        
        if path == "/api/status":
            self.send_json(session_manager.get_status())
        
        elif path == "/api/next":
            prompt = session_manager.get_current_prompt()
            if prompt is None:
                self.send_json({"done": True, "status": session_manager.get_status()})
            else:
                # Include cached responses if available
                cached = session_manager.get_cached_responses(prompt["prompt_index"])
                self.send_json({
                    "done": False, 
                    "prompt": prompt,
                    "cached_responses": {str(k): v for k, v in cached.items()}
                })
        
        elif path.startswith("/api/prompt/"):
            # Get prompt at specific index
            try:
                index = int(path.split("/")[-1])
                prompt = session_manager.get_prompt_at_index(index)
                if prompt is None:
                    self.send_json({"error": "Invalid prompt index"}, 400)
                else:
                    cached = session_manager.get_cached_responses(index)
                    self.send_json({
                        "prompt": prompt,
                        "cached_responses": {str(k): v for k, v in cached.items()}
                    })
            except ValueError:
                self.send_json({"error": "Invalid index"}, 400)
        
        elif path == "/api/cached":
            # Check if response is cached
            prompt_idx = int(query.get("prompt_idx", [session_manager.current_prompt_index])[0])
            temp_idx = int(query.get("temp_idx", [0])[0])
            is_cached = session_manager.has_cached_response(prompt_idx, temp_idx)
            cached_response = None
            if is_cached:
                cached_response = session_manager.response_cache[prompt_idx][temp_idx]
            self.send_json({
                "cached": is_cached,
                "response": cached_response
            })
        
        elif path == "/api/generate":
            # SSE endpoint for streaming generation
            temp_index = int(query.get("temp_index", [0])[0])
            temperature = float(query.get("temperature", [0.7])[0])
            
            prompt_data = session_manager.get_current_prompt()
            if prompt_data is None:
                self.send_json({"error": "No current prompt"}, 400)
                return
            
            # Set SSE headers
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            full_text = ""
            min_temp = None
            avg_temp = None
            try:
                for chunk in generate_streaming(
                    prompt_data["instruction"],
                    temperature,
                    max_length=global_args.max_length if global_args else 32768
                ):
                    if "error" in chunk:
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        break
                    
                    # Filter replacement character (U+FFFD) from token
                    if "token" in chunk:
                        token = chunk["token"]
                        # Remove unicode replacement character
                        if '\ufffd' in token:
                            print(f"[DEBUG] U+FFFD in SSE token: {repr(token)}")
                            token = token.replace('\ufffd', '')
                            chunk["token"] = token
                    
                    if not chunk.get("done", False):
                        full_text += chunk.get("token", "")
                    else:
                        # Extract temp info from done chunk
                        min_temp = chunk.get("min_temp_used")
                        avg_temp = chunk.get("avg_temp_used")
                    
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                
                # Store the completed response with temp info
                session_manager.store_response(
                    temp_index, full_text, temperature,
                    min_temp_used=min_temp,
                    avg_temp_used=avg_temp
                )
                
            except Exception as e:
                import traceback
                print(f"[ERROR] Generation failed: {e}")
                traceback.print_exc()
                error_data = {"error": str(e), "done": True}
                self.wfile.write(f"data: {json.dumps(error_data)}\n\n".encode())
        
        elif path == "/api/download":
            if session_manager.preferences_file and os.path.exists(session_manager.preferences_file):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Disposition", 
                    f"attachment; filename=preferences_{session_manager.session_id}.json")
                self.end_headers()
                with open(session_manager.preferences_file, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_json({"error": "No preferences file yet"}, 404)
        
        elif path == "/" or path == "/index.html":
            self.serve_static_file(os.path.join(STATIC_DIR, "index.html"))
        
        elif path.startswith("/"):
            # Serve static files
            filepath = os.path.join(STATIC_DIR, path.lstrip("/"))
            self.serve_static_file(filepath)
    
    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}
        
        if path == "/api/save":
            chosen_idx = data.get("chosen_idx")
            rejected_idx = data.get("rejected_idx")
            
            if chosen_idx is not None and rejected_idx is not None:
                success = session_manager.save_preference(int(chosen_idx), int(rejected_idx))
                self.send_json({
                    "success": success,
                    "status": session_manager.get_status()
                })
            else:
                self.send_json({"error": "Invalid choice"}, 400)
        
        elif path == "/api/skip":
            success = session_manager.skip_current()
            self.send_json({
                "success": success,
                "status": session_manager.get_status()
            })
        
        elif path == "/api/clear":
            # Clear preference for a specific prompt
            index = data.get("index", session_manager.current_prompt_index)
            success = session_manager.clear_preference(int(index))
            prompt = session_manager.get_current_prompt()
            self.send_json({
                "success": success,
                "prompt": prompt,
                "status": session_manager.get_status()
            })
        
        elif path == "/api/clear_all":
            # Clear all preferences
            success = session_manager.clear_all_preferences()
            prompt = session_manager.get_current_prompt()
            self.send_json({
                "success": success,
                "prompt": prompt,
                "status": session_manager.get_status()
            })
        
        elif path == "/api/delete_session":
            # Delete session files and exit
            deleted = session_manager.delete_session()
            self.send_json({
                "success": True,
                "deleted": deleted
            })
        
        elif path == "/api/prev":
            success = session_manager.go_prev()
            prompt = session_manager.get_current_prompt() if success else None
            cached = session_manager.get_cached_responses(session_manager.current_prompt_index) if success else {}
            self.send_json({
                "success": success,
                "prompt": prompt,
                "cached_responses": {str(k): v for k, v in cached.items()},
                "status": session_manager.get_status()
            })
        
        elif path == "/api/next":
            # POST /api/next: Navigate to next prompt (without saving preference)
            success = session_manager.go_next()
            prompt = session_manager.get_current_prompt() if success else None
            cached = session_manager.get_cached_responses(session_manager.current_prompt_index) if success else {}
            self.send_json({
                "success": success,
                "prompt": prompt,
                "cached_responses": {str(k): v for k, v in cached.items()},
                "status": session_manager.get_status()
            })
        
        elif path == "/api/goto":
            index = data.get("index")
            if index is not None:
                success = session_manager.go_to_prompt(int(index))
                prompt = session_manager.get_current_prompt() if success else None
                cached = session_manager.get_cached_responses(int(index)) if success else {}
                self.send_json({
                    "success": success,
                    "prompt": prompt,
                    "cached_responses": {str(k): v for k, v in cached.items()},
                    "status": session_manager.get_status()
                })
            else:
                self.send_json({"error": "Index required"}, 400)
        
        elif path == "/api/heartbeat":
            global last_heartbeat
            last_heartbeat = time.time()
            self.send_json({"ok": True})
        
        elif path == "/api/disconnect":
            global browser_connected
            browser_connected = False
            self.send_json({"ok": True})
        
        else:
            self.send_json({"error": "Not found"}, 404)
    
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def find_incomplete_sessions():
    """Find session files with pending prompts in temp_data directory."""
    sessions = []
    if not os.path.exists(TEMP_DATA_DIR):
        return sessions
    
    for f in os.listdir(TEMP_DATA_DIR):
        # Pattern: RLHF_preference_data_*_session.json
        if f.startswith("RLHF_preference_data_") and f.endswith("_session.json"):
            path = os.path.join(TEMP_DATA_DIR, f)
            try:
                with open(path, "r", encoding="utf-8") as file:
                    state = json.load(file)
                    remaining = state["progress"]["total"] - state["progress"]["completed"]
                    if remaining > 0:
                        sessions.append({
                            "path": path,
                            "id": state["session_id"],
                            "remaining": remaining,
                            "total": state["progress"]["total"],
                            "completed": state["progress"]["completed"]
                        })
            except (json.JSONDecodeError, KeyError):
                continue
    
    return sorted(sessions, key=lambda x: x["id"], reverse=True)


def select_session_interactive(sessions):
    """Display interactive session selection menu."""
    import msvcrt  # Windows keyboard input
    
    # Generate options list
    options = ["[New Session] Start fresh"]
    for sess in sessions:
        options.append(f"{sess['id']} ({sess['completed']}/{sess['total']} completed, {sess['remaining']} remaining)")
    
    current_index = 0
    
    def render():
        # Clear screen and show menu
        os.system('cls' if os.name == 'nt' else 'clear')
        print("[INFO] Found incomplete session(s):\n")
        print("Please choose a session (Up/Down to move, Enter to select):\n")
        for i, opt in enumerate(options):
            if i == current_index:
                print(f"  > {opt}")
            else:
                print(f"    {opt}")
        print()
    
    # Clear any pending input
    while msvcrt.kbhit():
        msvcrt.getch()
    
    render()
    
    while True:
        try:
            key = msvcrt.getch()
            
            if key == b'\r':  # Enter
                break
            elif key == b'\xe0':  # Arrow key prefix on Windows
                key2 = msvcrt.getch()
                if key2 == b'H':  # Up arrow
                    current_index = max(0, current_index - 1)
                elif key2 == b'P':  # Down arrow
                    current_index = min(len(options) - 1, current_index + 1)
                render()
            elif key == b'\x03':  # Ctrl+C
                print("\n[INFO] Cancelled")
                sys.exit(0)
        except KeyboardInterrupt:
            print("\n[INFO] Cancelled")
            sys.exit(0)
    
    if current_index == 0:
        return None  # New session
    else:
        return sessions[current_index - 1]["path"]


def find_model_path(model_name):
    """Find model path from model name."""
    base_dir = "model"
    path1 = os.path.join(base_dir, model_name)
    if os.path.exists(path1):
        return path1
    path2 = os.path.join(base_dir, "train", model_name)
    if os.path.exists(path2):
        return path2
    return None


def resolve_data_path(data_path):
    """Resolve data path, defaulting to data/ folder if not found."""
    # If path exists as-is, use it
    if os.path.exists(data_path):
        return data_path
    
    # Try prepending data/ folder
    data_folder_path = os.path.join("data", data_path)
    if os.path.exists(data_folder_path):
        return data_folder_path
    
    # Return original path (will raise error in load_prompts)
    return data_path


def load_prompts(data_path):
    """Load prompts from JSON file."""
    # Resolve path (check data/ folder if not found)
    resolved_path = resolve_data_path(data_path)
    
    with open(resolved_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "data" in data:
        return data["data"]
    else:
        return [data]


def load_model_and_tokenizer(args):
    """Load model and tokenizer into global variables."""
    global global_model, global_tokenizer, global_system_prompt
    
    print("[INFO] Loading model...")
    
    model_path = find_model_path(args.model)
    if model_path is None:
        raise FileNotFoundError(f"Model '{args.model}' not found in model/ or model/train/")
    print(f"[INFO] Model path: {model_path}")
    
    # Create args for model loading
    class ModelArgs:
        def __init__(self, model_type, model_path):
            self.model_type = model_type
            self.model_path = model_path
            self.load_until_layer = None
            self.freeze_until_layer = None
            self.base_model_path = None
            self.hidden_size = None
            self.num_hidden_layers = None
            self.num_attention_heads = None
            self.num_key_value_heads = None
            self.intermediate_size = None
            self.vocab_size = None
            self.max_position_embeddings = None
            self.rope_theta = None
            self.torch_dtype = None
    
    model_args = ModelArgs(args.model, model_path)
    global_model = model_module.get_model(model_args)
    global_model.eval()
    
    # Wrap with DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using DataParallel with {torch.cuda.device_count()} GPUs")
        global_model = nn.DataParallel(global_model)
    
    print(f"[INFO] Model loaded on {get_model_device(global_model)}")
    
    # Load tokenizer
    print("[INFO] Loading tokenizer...")
    tok_manager = TokenizerManager(model_type=args.model, tokenizer_base_dir="model")
    global_tokenizer = tok_manager.load_tokenizer()
    global_tokenizer.pad_token = global_tokenizer.eos_token
    
    # Load chat template
    file_config = get_file_config(args.model)
    if file_config:
        chat_template_path = os.path.join(file_config.BASE_PATH, file_config.CHAT_TEMPLATE)
        if os.path.exists(chat_template_path):
            with open(chat_template_path, "r", encoding="utf-8") as f:
                global_tokenizer.chat_template = f.read().strip()
        
        # Load system prompt: prompts folder (priority) > FileConfig path
        system_prompt_file = file_config.SYSTEM_PROMPT if hasattr(file_config, 'SYSTEM_PROMPT') else "SYSTEM_PROMPT.txt"
        prompts_path = os.path.join(os.path.dirname(__file__), "prompts", system_prompt_file)
        if os.path.exists(prompts_path):
            with open(prompts_path, "r", encoding="utf-8") as f:
                global_system_prompt = f.read().strip()
        else:
            system_prompt_path = os.path.join(file_config.BASE_PATH, system_prompt_file)
            if os.path.exists(system_prompt_path):
                with open(system_prompt_path, "r", encoding="utf-8") as f:
                    global_system_prompt = f.read().strip()
    
    print("[INFO] Tokenizer loaded")


def run_server(args, is_first_run=True):
    """Main server entry point."""
    global session_manager, global_args, global_server
    global_args = args
    
    # Check for incomplete sessions
    incomplete_sessions = find_incomplete_sessions()
    selected_path = None
    
    if incomplete_sessions:
        selected_path = select_session_interactive(incomplete_sessions)
        
        if selected_path:
            print(f"\n[INFO] Resuming session from {selected_path}")
            session_manager.load_session(selected_path)
            status = session_manager.get_status()
            print(f"[INFO] Loaded session: {status['completed']}/{status['total']} completed, "
                  f"{status['remaining']} remaining")
    else:
        # No incomplete sessions
        if not args.data_path or not args.model:
            print("[INFO] No incomplete sessions found and no data_path/model provided.")
            print("[INFO] Exiting. Use --data_path and --model to start a new session.")
            return False  # Exit without restart
        
        # Only ask for confirmation on restart (not first run)
        if not is_first_run:
            print("\n[INFO] No incomplete sessions found.")
            print(f"[INFO] Start new session with {args.data_path}? (Enter to continue, Ctrl+C to exit)")
            try:
                import msvcrt
                while msvcrt.kbhit():
                    msvcrt.getch()  # Clear input buffer
                msvcrt.getch()  # Wait for any key
            except KeyboardInterrupt:
                print("\n[INFO] Cancelled")
                return False
    
    if not selected_path:
        # New session
        if not args.data_path or not args.model:
            print("[ERROR] --data_path and --model are required for new sessions")
            print("Usage: python rlhf_collect.py --data_path data/sample.json --model ministral_3_3b_instruct")
            sys.exit(1)
        
        print(f"\n[INFO] Starting new session")
        
        # Resolve data path (defaults to data/ folder)
        resolved_data_path = resolve_data_path(args.data_path)
        print(f"[INFO] Data path: {resolved_data_path}")
        print(f"[INFO] Model: {args.model}")
        
        # Load prompts
        prompts = load_prompts(resolved_data_path)
        print(f"[INFO] Loaded {len(prompts)} prompts")
        
        # Parse temperatures
        temperatures = [float(t) for t in args.temperatures.split(",")]
        print(f"[INFO] Temperatures: {temperatures}")
        
        # Create session
        config = {
            "data_path": resolved_data_path,
            "model": args.model,
            "temperatures": temperatures,
            "max_length": args.max_length
        }
        session_manager.create_new_session(config, prompts, temperatures)
        status = session_manager.get_status()
        print(f"[INFO] Session created: {status['total']} prompts to label")
    
    # Start HTTP server
    url = f"http://localhost:{PORT}"
    print(f"\n[INFO] Starting server at {url}")
    print("[INFO] Model ready for streaming generation")
    print("[INFO] Press Ctrl+C to stop\n")
    
    # Open browser automatically
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    
    # Reset connection tracking
    global last_heartbeat, browser_connected
    last_heartbeat = time.time()
    browser_connected = True
    
    # Heartbeat watcher thread
    def heartbeat_watcher(httpd):
        global browser_connected
        while browser_connected:
            time.sleep(2)
            if time.time() - last_heartbeat > HEARTBEAT_TIMEOUT:
                print("\n[INFO] Browser connection lost (heartbeat timeout). Returning to session selection...")
                browser_connected = False
                httpd.shutdown()
                return
        
        # beforeunload triggered
        print("\n[INFO] Browser closed. Returning to session selection...")
        time.sleep(0.5)  # Brief delay to ensure response is sent
        httpd.shutdown()
    
    try:
        global_server = ThreadingHTTPServer(("localhost", PORT), PreferenceHandler)
        
        # Start heartbeat watcher
        watcher = threading.Thread(target=heartbeat_watcher, args=(global_server,), daemon=True)
        watcher.start()
        
        global_server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped by user")
        status = session_manager.get_status()
        print(f"[INFO] Final status: {status['completed']} completed")
        if session_manager.preferences_file and os.path.exists(session_manager.preferences_file):
            size = os.path.getsize(session_manager.preferences_file)
            print(f"[INFO] Preferences saved to: {session_manager.preferences_file} ({size} bytes)")
        if status['remaining'] > 0:
            print(f"[INFO] {status['remaining']} prompts remaining. Run again to continue.")
        return False  # Don't restart on manual stop
    except OSError as e:
        if "Address already in use" in str(e) or "10048" in str(e):
            print(f"[ERROR] Port {PORT} is already in use.")
        else:
            raise
        return False
    finally:
        # Clean up server only (keep model in memory for next session)
        cleanup_server()
    
    return True  # Restart session selection


if __name__ == "__main__":
    # Check for detailed help (--arg_name --help)
    from utils.detailed_help import check_detailed_help
    check_detailed_help()
    
    parser = argparse.ArgumentParser(description="RLHF Preference Data Collection Tool (Streaming)")
    
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to prompts JSON file (required for new session)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name or path (required for new session)")
    parser.add_argument("--temperatures", type=str, default="0.5,0.7,1.0,1.2",
                        help="Comma-separated temperatures for response generation (default: 0.5,0.7,1.0,1.2)")
    parser.add_argument("--max_length", type=int, default=32768,
                        help="Max new tokens to generate (default: 32768)")
    
    # Refusal mechanism parameters (same as inference.py)
    parser.add_argument("--no_refusal", action="store_true",
                        help="Disable refusal mechanism")
    parser.add_argument("--refusal_threshold", type=float, default=3.0,
                        help="Uncertainty threshold for refusal (std of logits, default: 3.0)")
    parser.add_argument("--refusal_max_retries", type=int, default=3,
                        help="Max retries per token when refused (default: 3)")
    parser.add_argument("--refusal_temp_decay", type=float, default=0.8,
                        help="Temperature multiplier on each retry (default: 0.8)")
    parser.add_argument("--refusal_min_temp", type=float, default=0.4,
                        help="Minimum temperature for refusal mechanism (default: 0.4)")
    parser.add_argument("--refusal_recovery_tokens", type=int, default=3,
                        help="Number of tokens to recover to original temperature after refusal (default: 3)")
    parser.add_argument("--refusal_recovery_method", type=str, default="exponential",
                        choices=["linear", "exponential", "ease_out", "ease_in_out", "step"],
                        help="Temperature recovery curve after refusal (default: exponential)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter (default: 50)")
    
    # Environment selection
    parser.add_argument("--local", action="store_true",
                        help="Use local paths instead of server paths (default: server). Also sets GPU default to 0.")
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU IDs to use (e.g., '0' or '6,7'). Default: '6,7' (server) or '0' (local)")
    
    args = parser.parse_args()
    
    # Set environment based on --local flag
    global TEMP_DATA_DIR, FINAL_DATA_DIR
    set_local_mode(args.local)
    TEMP_DATA_DIR = get_temp_data_dir()
    FINAL_DATA_DIR = get_data_dir()
    ensure_dirs()
    
    # Kill Node.js processes to prevent GPU conflicts
    kill_nodejs_processes()
    
    # Register cleanup handlers once at startup
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("[WARNING] CUDA not available. Generation will be slow.")
    else:
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model once at startup (kept in memory across sessions)
    try:
        load_model_and_tokenizer(args)
    except Exception as e:
        print(f"\n[ERROR] Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("[INFO] Model loaded and ready")
    
    # Session selection loop - restarts when browser closes
    is_first_run = True
    while True:
        should_restart = run_server(args, is_first_run)
        is_first_run = False  # After first run
        if not should_restart:
            break
        print("\n" + "="*50)
        print("[INFO] Restarting session selection...")
        print("="*50 + "\n")
