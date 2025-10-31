# NOEMA — اسکلت «مغز نوما» (نسخه‌ی V0)

> یک اسکلت سبک و ماژولار برای شروع ساخت «نوما»: عامل گفت‌وگوگر با چرخه‌ی حس–عمل،
> ایمنی پایه، مهارت‌ها، رجیستری ابزار، «خواب/تثبیت» آفلاین و مدلِ‌خود (Self-Model).

---

## ✨ چه‌چیزی آماده است؟

- **I/O متنی**: دریافت/ارسال پیام، ثبت اپیزودها (JSONL یا EpisodeStore اختیاری).
- **لایه‌ی زبان**: تشخیص نیت‌های ساده (`greeting`, `compute`) + قالب‌بندی پاسخ.
- **مهارت‌ها (Skills)**: `reply_greeting` و `invoke_calc` (ماشین‌حساب امن).
- **ToolHub**: رجیستری ابزارها + اعتبارسنجی آرگومان‌ها (YAML/JSON).
- **ایمنی (Safety)**: DSL قواعد ایمنی + «سپر زمان‌اجرا» برای فیلتر کردن کنش‌ها.
- **Self-Model**: بردار وضعیت درونی + کالیبراسیون اعتماد آنلاین.
- **Sleep/Offline**: چرخه‌ی «خواب/تثبیت» (بازسازی ایندکس، مفاهیم، گراف، کالیبراسیون).
- **تست‌های دودکشی**: ۲ تست رگرسیونی نمونه (`arith-2plus2`, `style-greeting`).
- **اسکریپت راه‌انداز**: ایجاد پوشه‌ها/پیکربندی‌های لازم.

> این نسخه «آموزشی/مفهومی» است تا سریع راه بیفتید؛ بعداً می‌توانید بخش‌ها را عوض/گسترش دهید.

---

## 📦 نصب و راه‌اندازی

### 1) پیش‌نیاز
- Python 3.10+
- (اختیاری) `faiss-cpu`, `scikit-learn`, `pandas`, `pyarrow` برای قابلیت‌های افزوده

### 2) کلون و نصب محلی
```bash
git clone <your-repo-url> noema
cd noema
python -m venv .venv && source .venv/bin/activate   # ویندوز: .venv\Scripts\activate
pip install -e .

3) آماده‌سازی پوشه‌ها/فایل‌ها
python scripts/init.py

4) اجرای تست‌های نمونه
python -m tests.test_runner
# یا
python tests/test_runner.py --pattern tests/regression/*.yaml

🚀 اجرای نمونهٔ تعاملی (Minimal Loop)

فایل زیر را به‌صورت موقت اجرا کنید تا چرخه‌ی کوچک گفت‌وگو را ببینید:

# demo_minimal.py
from lang import parse, load_style, format_reply
from skills.invoke_calc import run as run_calc
from skills.reply_greeting import run as run_greet
from env import make_text_env
from safety import load_rules, enforce
from world import Action if False else None  # فقط برای تایپ‌هینت

rules = load_rules()             # از config/safety.yaml
env = make_text_env()            # لاگ به data/episodes/episodes.jsonl
style = load_style()

print("نوما آماده‌ست. بنویس: سلام یا یک عبارت مثل 7*(5-2)")
while True:
    try:
        user = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        break
    if not user:
        continue

    env.begin_turn(user)
    plan = parse(user)  # {"intent": "...", "args": {...}, "confidence": ...}

    if plan["intent"] == "greeting":
        out = run_greet(user_text=user, plan=plan, style=style)
        action = {"kind":"skill","name":"reply_greeting","args":{}}
    elif plan["intent"] == "compute":
        out = run_calc(user_text=user, plan=plan, style=style)
        action = {"kind":"tool","name":"invoke_calc","args": {"expr": out["outcome"].get("expr","")}}
    else:
        text = format_reply(intent="unknown", outcome={}, style=style, meta={"confidence":0.4})
        out = {"intent":"unknown","outcome":{},"text_out":text,"meta":{"confidence":0.4}}
        action = {"kind":"policy","name":"ask_clarify","args":{}}

    # اعمال سپر ایمنی (در حالت review/block → clarify)
    from safety import enforce
    safe_action, decision = enforce(
        text=user,
        plan=plan,
        action=type("A",(object,),{"kind":action["kind"],"name":action["name"],"args":action["args"]})(),
        state={"u": 1.0 - out["meta"].get("confidence", 0.0), "conf": out["meta"].get("confidence", 0.0)},
        rules=rules,
    )

    # خروجی + ثبت اپیزود
    step = env.deliver(
        intent=out["intent"],
        action={"kind":safe_action.kind,"name":safe_action.name,"args":safe_action.args},
        text_out=out["text_out"],
        meta=out["meta"],
        feedback=None,  # بعداً مربی می‌تواند -1/0/+1 بدهد
        label_ok=out.get("label_ok"),
    )
    print(step.text_out)


اجرا:

python demo_minimal.py

🧭 ساختار پوشه‌ها
config/
  safety.yaml         # DSL قواعد ایمنی
  tools.yaml          # رجیستری ابزارها
  value.yaml          # وزن‌دهی ارزش/پاداش
  meta.yaml           # سبک زبان و تنظیمات کنترل
control/              # (رزرو) کنترل‌گر/برنامه‌ریز — بعداً پر می‌شود
env/
  io_text.py          # محیط I/O متنی + لاگ اپیزود
  __init__.py
lang/
  parse.py            # تشخیص نیت/آرگومان
  format.py           # قالب‌بندی پاسخ
  __init__.py
safety/
  dsl.py              # بارگذار/ارزیاب قواعد
  shield.py           # سپر زمان‌اجرا
  __init__.py
selfmeta/
  self_model.py       # مدلِ خود + بردار وضعیت
  calibrate.py        # کالیبراسیون اعتماد (آنلاین، بنینگ)
  __init__.py
sleep/
  offline.py          # چرخه‌ی «خواب/تثبیت» آفلاین
  __init__.py
skills/
  manifest.yaml       # فهرست مهارت‌ها
  reply_greeting.py   # مهارت سلام
  invoke_calc.py      # مهارت ماشین‌حساب امن
  __init__.py
toolhub/
  registry.py         # رجیستری ابزارها
  verify.py           # اعتبارسنج آرگومان
  __init__.py
tests/
  regression/*.yaml   # تست‌های دودکشی
  test_runner.py
scripts/
  init.py             # آماده‌سازی پوشه‌ها/پیکربندی‌ها
  init_faiss.py       # ساخت ایندکس (اختیاری)
  migrate_parquet.py  # تبدیل JSONL→Parquet (اختیاری)
models/               # (رزرو) وزن/مدل‌ها

🧩 معماری مفهومی (۱۰ بلوک)

Perception (lang/parse + env)

World Model (رزرو)

Memory (رزرو)

Attention/Controller (control/ رزرو)

Motivation/Value (config/value.yaml + لاگ اپیزود)

Policy/Planner (رزرو)

Self-Model (selfmeta/*)

Uncertainty & Safety (selfmeta, safety/*)

Sleep/Consolidation (sleep/offline.py)

Grounded Language/Skills (lang/*, skills/*, toolhub/*)

در V0 تمرکز روی «ستون فقرات» است: I/O متنی، نیت‌های ساده، مهارت‌ها، ایمنی، مدلِ‌خود و چرخه‌ی خواب.

🧪 قرارداد خروجی مهارت‌ها

هر مهارت باید dict زیر را برگرداند:

{
  "intent": "<greeting|compute|...>",
  "outcome": {...},              # دادهٔ ساخت‌یافته برای lang/format
  "text_out": "<string>",        # متن نهایی برای نمایش
  "meta": { "confidence":0.9, "u":0.1, "r_total":0.0, "risk":0.0 },
  "extras": {...},               # اختیاری
  "label_ok": True/False         # اگر ارزیابی/اجرا موفق نبود → False
}

🛡️ ایمنی و قیود

قواعد را در config/safety.yaml تعریف/ویرایش کنید.

«سپر» با توجه به intent, action, text, conf/u تصمیم allow/review/block می‌دهد.

در حالت review/block، به‌صورت پیش‌فرض کنش «clarify» جایگزین می‌شود.

📈 کالیبراسیون اعتماد

فایل data/calibration.json توسط selfmeta/calibrate.py نگهداری می‌شود.

چرخه‌ی خواب (sleep/offline.py) از اپیزودها جفت‌های (p_raw, y) جمع‌آوری کرده و کالیبره می‌کند.

اگر label_ok را هنگام آموزش پر کنید، کالیبراسیون دقیق‌تر می‌شود.

🌙 چرخهٔ خواب

برای اجرای کامل (در صورت آماده بودن ماژول‌های اختیاری memory/concept):

python -m sleep.init
# یا با پارامتر:
python -m sleep.offline --episodes data/episodes --dim 64

🧰 توسعهٔ مهارت جدید

یک فایل مثل skills/my_skill.py با تابع run(...) بسازید.

در skills/manifest.yaml مهارت را اضافه کنید (نام، kind، entry، allowed_args).

در لایه‌ی بالا (app/control) با توجه به intent آن را فراخوانی کنید.

🗺️ نقشهٔ راه (پیشنهادی)

V1: افزودن memory/* (EpisodeStore کامل، بردارها، FAISS)، concept/* (خوشه‌بندی + گراف)،
control/* (برنامه‌ریز کوتاه‌برد با ارزش ترکیبی)، گسترش نیت‌ها/مهارت‌ها.

V2: مدلِ جهان نهان، کنجکاوی مفید، مهارت‌های سلسله‌مراتبی، بازیابی اپیزودیک مؤثر.

V3: اتصال زبان به عمل چندحسی، ایمنی پیشرفته، یادگیری پیوسته مقاوم به تداخل.

❓ عیب‌یابی سریع

تست‌ها ران نمی‌شوند: مطمئن شوید PyYAML نصب است یا فایل‌های YAML به‌درستی فرمت شده‌اند.

لاگ اپیزود نوشته نمی‌شود: پوشه‌ی data/episodes/ را بسازید یا scripts/init.py را اجرا کنید.

محاسبه رد می‌شود: عبارت فقط باید شامل 0-9, +, -, *, /, (, ) باشد.

خطای ماژول‌های اختیاری: بخش‌های مربوطه را غیرفعال کنید یا بسته‌های اختیاری را نصب کنید.

📜 مجوز

MIT — آزاد برای استفاده/تغییر با ذکر نام.

🤝 سپاس

این اسکلت برای یادگیری و ساخت تدریجی «نوما» طراحی شده است: ساده اما محکم.
هر بخش را که آماده بودید، عوض/تقویت کنید—اما ثبت اپیزودها، ایمنی و کالیبراسیون را
از همان روز اول فعال نگه دارید.