try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Bot = None

import asyncio
import json
import os
import sys

def load_config():
    """Load Telegram bot configuration"""
    config_path = "telegram_config.json"
    if not os.path.exists(config_path):
        return None
    
    with open(config_path, "r") as f:
        return json.load(f)

def format_attendance_message(name, action, time, confidence):
    """Format attendance message for Telegram"""
    status = "Entry" if action == "Entry" else "Exit"
    return f"Name: {name}\nTime: {time}\nStatus: {status}"

class TelegramNotifier:
    def __init__(self, token, chat_id):
        if not TELEGRAM_AVAILABLE:
            raise ImportError("Telegram module not available")
        self.bot = Bot(token=token)
        self.chat_id = chat_id
        self._loop = None
    
    def _get_or_create_loop(self):
        """Get existing event loop or create a new one"""
        try:
            # Try to get the running loop
            loop = asyncio.get_running_loop()
            return loop, False  # Return existing loop, don't close it
        except RuntimeError:
            # No running loop, create a new one
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            return self._loop, True  # Return new loop, can close it
    
    def send_sync(self, message):
        """Send message synchronously - SIMPLIFIED VERSION"""
        try:
            # Force use of subprocess to avoid event loop conflicts
            import subprocess
            import tempfile
            
            # Create a simple Python script to send the message
            script = f'''
import asyncio
from telegram import Bot

async def send():
    bot = Bot(token="{self.bot.token}")
    await bot.send_message(chat_id="{self.chat_id}", text="""{message}""")

asyncio.run(send())
'''
            
            # Write script to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script)
                script_path = f.name
            
            try:
                # Run the script in a subprocess with timeout
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    print(f"[TELEGRAM] ✅ Message sent successfully")
                else:
                    print(f"[TELEGRAM] ❌ Send failed: {result.stderr}")
            finally:
                # Clean up temp file
                try:
                    os.unlink(script_path)
                except:
                    pass
                    
        except Exception as e:
            print(f"[TELEGRAM ERROR] Failed to send message: {e}")
            import traceback
            traceback.print_exc()

def send_attendance_notification(name, action, time, confidence):
    """Send attendance notification via Telegram"""
    if not TELEGRAM_AVAILABLE:
        print("[TELEGRAM] Module not available")
        return
    
    config = load_config()
    if not config or not config.get("enabled", False):
        print("[TELEGRAM] Not enabled in config")
        return
    
    token = config.get("bot_token") or config.get("token")
    chat_id = config.get("chat_id")
    
    if not token or not chat_id:
        print("[TELEGRAM] Bot not configured properly")
        return
    
    try:
        notifier = TelegramNotifier(token, chat_id)
        message = format_attendance_message(name, action, time, confidence)
        notifier.send_sync(message)
    except Exception as e:
        print(f"[TELEGRAM ERROR] Error sending notification: {e}")
        import traceback
        traceback.print_exc()
