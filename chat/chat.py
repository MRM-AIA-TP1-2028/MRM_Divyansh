"""LAN Chat (terminal version) - run `python chat.py` on every device. One device chooses option 1 (host)."""
import hashlib, hmac, json, os, queue, socket, sqlite3, threading, time

PORT = 5000

# ------------------------------- SERVER -------------------------------
db = sqlite3.connect("chat.db", check_same_thread=False)
db.executescript("CREATE TABLE IF NOT EXISTS users(name TEXT PRIMARY KEY COLLATE NOCASE, salt BLOB, hash BLOB);"
                 "CREATE TABLE IF NOT EXISTS msgs(src TEXT, dst TEXT, body TEXT, ts REAL);")
lock, online = threading.Lock(), {}          # online: socket -> username

def hp(pw, salt): return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 100_000)

def tx(s, obj):
    try: s.sendall((json.dumps(obj) + "\n").encode())
    except OSError: pass

def push_users():                            # call with lock held
    on = set(online.values())
    msg = {"type": "users", "users": [[n, n in on] for (n,) in db.execute("SELECT name FROM users ORDER BY name")]}
    for c in online: tx(c, msg)

def serve_client(s):
    user = None
    try:
        for line in s.makefile("r", encoding="utf-8"):
            m = json.loads(line); a = m["action"]
            with lock:
                if a == "register":
                    salt = os.urandom(16)
                    try:
                        db.execute("INSERT INTO users VALUES(?,?,?)", (m["user"].strip(), salt, hp(m["pw"], salt))); db.commit()
                        tx(s, {"type": "info", "text": "Registered! Now log in."}); push_users()
                    except sqlite3.IntegrityError:
                        tx(s, {"type": "error", "text": "Username already taken"})
                elif a == "login":
                    r = db.execute("SELECT name, salt, hash FROM users WHERE name=?", (m["user"],)).fetchone()
                    if r and hmac.compare_digest(hp(m["pw"], r[1]), r[2]):
                        user = online[s] = r[0]; tx(s, {"type": "login", "user": user}); push_users()
                    else:
                        tx(s, {"type": "error", "text": "Wrong username or password"})
                elif user and a == "history":
                    rows = db.execute("SELECT src,dst,body,ts FROM msgs WHERE (src=? AND dst=?) OR (src=? AND dst=?) ORDER BY ts",
                                      (user, m["with"], m["with"], user)).fetchall()
                    tx(s, {"type": "history", "with": m["with"], "rows": rows})
                elif user and a == "send":
                    dst = db.execute("SELECT name FROM users WHERE name=?", (m["to"],)).fetchone()
                    if dst and m["body"].strip():
                        msg = {"type": "msg", "src": user, "dst": dst[0], "body": m["body"].strip()[:2000], "ts": time.time()}
                        db.execute("INSERT INTO msgs VALUES(?,?,?,?)", (msg["src"], msg["dst"], msg["body"], msg["ts"])); db.commit()
                        for c, u in online.items():
                            if u in (user, dst[0]): tx(c, msg)
    except (OSError, ValueError, KeyError):
        pass
    finally:
        with lock:
            online.pop(s, None); push_users()
        s.close()

def start_server():
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT)); srv.listen()
    def loop():
        while True:
            threading.Thread(target=serve_client, args=(srv.accept()[0],), daemon=True).start()
    threading.Thread(target=loop, daemon=True).start()

def my_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try: s.connect(("10.255.255.255", 1)); return s.getsockname()[0]
    except OSError: return "127.0.0.1"

# ------------------------------- CLIENT (terminal) -------------------------------
sock = me = peer = None
users, q = [], queue.Queue()

def reader(s):
    try:
        for line in s.makefile("r", encoding="utf-8"): q.put(json.loads(line))
    except (OSError, ValueError): pass
    q.put({"type": "closed"})

def send(obj): sock.sendall((json.dumps(obj) + "\n").encode())

def fmt(src, body, ts):                      # src=None -> no name shown (chat with your own id)
    name = f"{src}: " if src else ""
    return f"[{time.strftime('%d %b %H:%M', time.localtime(ts))}] {name}{body}"

def printer():                               # prints everything the server pushes to us
    global users
    while True:
        m = q.get(); t = m["type"]
        if t == "users": users = m["users"]
        elif t == "history":
            print(f"\n--- chat with {m['with']} ({len(m['rows'])} earlier messages) ---")
            for r in m["rows"]: print(fmt(None if r[0] == r[1] else r[0], r[2], r[3]))
        elif t == "msg":
            own = m["src"] == m["dst"]           # message to your own id (same id, other terminal)
            if (m["dst"] if m["src"] == me else m["src"]) == peer: print(fmt(None if own else m["src"], m["body"], m["ts"]))
            elif own: print("\n** new message from your other terminal  (open it with /chat <your own id>) **")
            elif m["src"] == me: print(f"\n** your other terminal sent a message to {m['dst']} **")
            else: print(f"\n** new message from {m['src']}  (type /chat {m['src']} to open) **")
        elif t == "closed": print("\nDisconnected from server."); os._exit(0)

def main():
    global sock, me, peer
    mode = input("1) Host the server on this device\n2) Join a server on another device\nChoose 1 or 2: ").strip()
    if mode == "1":
        start_server(); ip = "127.0.0.1"
        print(f"Server running. On other devices choose 2 and enter IP: {my_ip()}")
    else:
        ip = input("Server IP: ").strip()
    try:
        sock = socket.create_connection((ip, PORT), timeout=5); sock.settimeout(None)
    except OSError as e:
        return print("Could not connect:", e)
    threading.Thread(target=reader, args=(sock,), daemon=True).start()
    while not me:                            # login / register
        act = "login" if input("\n(l)ogin or (r)egister? ").strip().lower().startswith("l") else "register"
        send({"action": act, "user": input("Username: ").strip(), "pw": input("Password: ")})
        while True:
            m = q.get()
            if m["type"] in ("error", "info"): print(m["text"]); break
            if m["type"] == "login": me = m["user"]; break
            if m["type"] == "closed": return print("Disconnected.")
    threading.Thread(target=printer, daemon=True).start()
    print(f"\nWelcome {me}!  Commands: /users  /chat <name>  /quit   (any other text is sent to the open chat)")
    while True:
        try: line = input().strip()
        except (EOFError, KeyboardInterrupt): return
        if not line: continue
        if line == "/quit": return
        if line == "/users":
            print("Users:", ", ".join(f"{n}{' (you)' if n == me else ''}{' (online)' if on else ''}" for n, on in users) or "none yet")
        elif line.startswith("/chat"):
            want = line[5:].strip().lower()
            match = [n for n, _ in users if n.lower() == want]
            if match: peer = match[0]; send({"action": "history", "with": peer})
            else: print("No such user. Type /users to see who is registered.")
        elif peer: send({"action": "send", "to": peer, "body": line})
        else: print("Open a chat first:  /chat <name>")

if __name__ == "__main__":
    main()