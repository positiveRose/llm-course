import sqlite3, json
db = sqlite3.connect('support.db')
rows = db.execute('SELECT session_id, seq, payload FROM messages ORDER BY session_id, seq').fetchall()
for sid, seq, payload in rows:
    data = json.loads(payload)
    print(f'[{sid[:8]}] seq={seq}: {json.dumps(data, ensure_ascii=False)[:200]}')
