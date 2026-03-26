import telebot
import os
import threading
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
alert_manager_ref = None

def send_alert_with_button(img_path, caption):
    if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE" or TELEGRAM_CHAT_ID == "YOUR_CHAT_ID_HERE":
        print("[Telegram Bot] Không thể gửi cảnh báo: Token hoặc Chat ID chưa được thiết lập.")
        return
        
    markup = InlineKeyboardMarkup()
    btn = InlineKeyboardButton(text="✅ Xác nhận (Acknowledge)", callback_data="ack_alert")
    markup.add(btn)
    
    try:
        with open(img_path, "rb") as photo:
            bot.send_photo(TELEGRAM_CHAT_ID, photo, caption=caption, reply_markup=markup)
    except Exception as e:
        print(f"[Telegram Bot] Lỗi khi gửi ảnh lên Telegram: {e}")

@bot.callback_query_handler(func=lambda call: call.data == 'ack_alert')
def ack_alert_callback(call):
    global alert_manager_ref
    if alert_manager_ref is not None:
        alert_manager_ref.acknowledge_alert()
        
    try:
        # Sửa tin nhắn gốc để xóa nút bấm và thêm trạng thái xác nhận
        new_caption = (call.message.caption or "") + "\n\n🟢 ĐÃ XÁC NHẬN (Snoozed)"
        bot.edit_message_caption(caption=new_caption, chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=None)
        
        # Trả lời lại callback query để dừng hiệu ứng vòng chờ loading trên app
        bot.answer_callback_query(call.id, "Đã xác nhận cảnh báo!")
    except Exception as e:
        print(f"[Telegram Bot] Lỗi khi cập nhật trạng thái tin nhắn gốc: {e}")

def start_bot_thread(manager_instance):
    global alert_manager_ref
    alert_manager_ref = manager_instance
    
    if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("[Telegram Bot] Token chưa cài đặt. Luồng chờ tin nhắn sẽ không được khởi động.")
        return
        
    polling_thread = threading.Thread(target=bot.infinity_polling, daemon=True)
    polling_thread.start()
    print("[Telegram Bot] Luồng kết nối trực tiếp BOT Telegram đã khởi động thành công.")
