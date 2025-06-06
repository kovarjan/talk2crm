# talk2api
talk2api - Natural language interface for controlling APIs via voice or text. .

# 🗣️ talk2api

**talk2api** is an open-source voice and natural language assistant that lets users interact with APIs — Currently supports CRM workflows — using simple voice commands.

💬 _“Add a new contact named Jan Novák from Acmark”_  
📇 _“Show deals closing this month.”_  
📞 _“Log a call with Petra about the marketing campaign.”_

---

## 🚀 Features

- 🎙️ Voice command recognition (supports Czech and English)
- 🤖 AI-powered intent detection (via NLP)
- 📇 CRM actions: create/update contacts, deals, meetings, etc.
- 🔗 API integration layer (pluggable)
- 🧪 Test suite with example commands for QA

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/talk2api.git
cd talk2api
pip install -r requirements.txt
```

## Start of the application

```bash
conda activate talk2api
# Ensure you have the correct environment activated
python main.py
uvicorn main:app --reload --port 3000 --host 0.0.0.
# audio recording websites must use https - use nginx proxy same origin
```
