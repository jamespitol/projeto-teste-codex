import csv
import io
import os
from datetime import datetime, timedelta
from typing import Optional

import google.generativeai as genai
import httpx
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jose import JWTError, jwt
from openai import OpenAI
from passlib.context import CryptContext
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "change_me")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./blog_manager.db")
MASTER_KEY_PATH = os.getenv("MASTER_KEY_PATH", ".key")

if not os.path.exists(MASTER_KEY_PATH):
    with open(MASTER_KEY_PATH, "wb") as f:
        f.write(Fernet.generate_key())
with open(MASTER_KEY_PATH, "rb") as f:
    fernet = Fernet(f.read())

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")


class User(Base):
    __tablename__ = "users"
    id = Integer(primary_key=True, index=True)
    username = String(255)
    password_hash = String(255)
    created_at = DateTime(default=datetime.utcnow)


class Blog(Base):
    __tablename__ = "blogs"
    id = Integer(primary_key=True, index=True)
    name = String(255)
    wp_url = String(255)
    wp_user = String(255)
    wp_app_password_encrypted = Text
    niche = String(255)
    default_prompt = Text
    is_active = Boolean(default=True)
    articles_per_week = Integer(default=1)
    preferred_days = String(64, default="0,2,4")
    preferred_time = String(8, default="09:00")
    created_at = DateTime(default=datetime.utcnow)
    articles = relationship("Article", back_populates="blog")


class Prompt(Base):
    __tablename__ = "prompts"
    id = Integer(primary_key=True, index=True)
    name = String(255)
    content = Text
    created_at = DateTime(default=datetime.utcnow)


class Setting(Base):
    __tablename__ = "settings"
    id = Integer(primary_key=True, index=True)
    openai_key_encrypted = Text
    gemini_key_encrypted = Text
    status_interval_minutes = Integer(default=30)


class Article(Base):
    __tablename__ = "articles"
    id = Integer(primary_key=True, index=True)
    title = String(255)
    keyword = String(255)
    content = Text
    ai_model = String(64)
    word_count = Integer(default=0)
    status = String(64, default="Aguardando revisão")
    category_name = String(255)
    wp_post_id = String(64, nullable=True)
    scheduled_for = DateTime(nullable=True)
    published_at = DateTime(nullable=True)
    created_at = DateTime(default=datetime.utcnow)
    image_url = Text(nullable=True)
    blog_id = Integer(ForeignKey("blogs.id"))
    blog = relationship("Blog", back_populates="articles")


Base.metadata.create_all(bind=engine)

app = FastAPI(title="Blog Manager")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
scheduler = BackgroundScheduler()


def db_dep():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def encrypt(text: str) -> str:
    return fernet.encrypt(text.encode()).decode()


def decrypt(token: str | None) -> str:
    if not token:
        return ""
    return fernet.decrypt(token.encode()).decode()


def create_access_token(data: dict):
    to_encode = data.copy()
    to_encode["exp"] = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(request: Request, db: Session = Depends(db_dep)):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401)
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user = db.get(User, int(payload.get("sub")))
        if not user:
            raise HTTPException(status_code=401)
        return user
    except (JWTError, ValueError):
        raise HTTPException(status_code=401)


def ensure_settings(db: Session):
    settings = db.query(Setting).first()
    if not settings:
        settings = Setting(status_interval_minutes=30)
        db.add(settings)
        db.commit()
        db.refresh(settings)
    return settings


def wp_headers(blog: Blog):
    auth = httpx.BasicAuth(blog.wp_user, decrypt(blog.wp_app_password_encrypted))
    return auth


def smart_slots(blog: Blog, qty: int, db: Session):
    days = [int(x) for x in blog.preferred_days.split(",") if x]
    hour, minute = map(int, blog.preferred_time.split(":"))
    slots = []
    now = datetime.now()
    dt = now
    existing = {a.scheduled_for for a in db.query(Article).filter(Article.blog_id == blog.id, Article.status == "Agendado").all() if a.scheduled_for}
    while len(slots) < qty:
        dt += timedelta(days=1)
        candidate = dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate.weekday() in days and candidate > now and candidate not in existing:
            slots.append(candidate)
    return slots


async def generate_text(model: str, prompt: str, openai_key: str, gemini_key: str):
    try:
        if model in {"gpt-4", "gpt-4o"}:
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
            return resp.choices[0].message.content
        genai.configure(api_key=gemini_key)
        m = genai.GenerativeModel("gemini-1.5-pro" if model == "gemini-pro" else "gemini-1.5-flash")
        return m.generate_content(prompt).text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro IA: {e}")


async def wp_post(blog: Blog, title: str, content: str, category_id: Optional[int], status_post: str = "draft", date_gmt: Optional[str] = None):
    payload = {"title": title, "content": content, "status": status_post}
    if category_id:
        payload["categories"] = [category_id]
    if date_gmt:
        payload["date_gmt"] = date_gmt
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.post(f"{blog.wp_url.rstrip('/')}/wp-json/wp/v2/posts", auth=wp_headers(blog), json=payload)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro WordPress: {e}")


def refresh_status_job():
    db = SessionLocal()
    try:
        for article in db.query(Article).filter(Article.wp_post_id.isnot(None)).all():
            blog = db.get(Blog, article.blog_id)
            if not blog:
                continue
            try:
                r = httpx.get(f"{blog.wp_url.rstrip('/')}/wp-json/wp/v2/posts/{article.wp_post_id}", auth=wp_headers(blog), timeout=20)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("status") == "publish":
                        article.status = "Publicado"
                        article.published_at = datetime.utcnow()
            except Exception:
                continue
        db.commit()
    finally:
        db.close()


@app.on_event("startup")
def startup():
    if not scheduler.running:
        scheduler.start()
    db = SessionLocal()
    interval = ensure_settings(db).status_interval_minutes
    db.close()
    scheduler.add_job(refresh_status_job, "interval", minutes=interval, id="status_refresh", replace_existing=True)


@app.get("/", response_class=HTMLResponse)
def root(request: Request, db: Session = Depends(db_dep)):
    if not db.query(User).first():
        return RedirectResponse("/setup")
    try:
        get_current_user(request, db)
    except HTTPException:
        return RedirectResponse("/login")
    counts = {s: db.query(Article).filter(Article.status == s).count() for s in ["Publicado", "Agendado", "Rascunho", "Aguardando revisão"]}
    upcoming = db.query(Article).filter(Article.status == "Agendado").order_by(Article.scheduled_for.asc()).limit(5).all()
    latest = db.query(Article).order_by(Article.created_at.desc()).limit(5).all()
    by_blog = db.query(Blog).all()
    blog_counts = [{"blog": b.name, "count": db.query(Article).filter(Article.blog_id == b.id).count()} for b in by_blog]
    return templates.TemplateResponse("dashboard.html", {"request": request, "counts": counts, "upcoming": upcoming, "latest": latest, "blog_counts": blog_counts})


@app.get("/setup", response_class=HTMLResponse)
def setup_page(request: Request, db: Session = Depends(db_dep)):
    if db.query(User).first():
        return RedirectResponse("/login")
    return templates.TemplateResponse("setup.html", {"request": request})


@app.post("/setup")
def setup_user(username: str = Form(...), password: str = Form(...), db: Session = Depends(db_dep)):
    if db.query(User).first():
        raise HTTPException(status_code=400, detail="Setup já concluído")
    user = User(username=username, password_hash=pwd_context.hash(password))
    db.add(user)
    db.commit()
    return RedirectResponse("/login", status_code=303)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
def login(username: str = Form(...), password: str = Form(...), db: Session = Depends(db_dep)):
    user = db.query(User).filter(User.username == username).first()
    if not user or not pwd_context.verify(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Credenciais inválidas")
    token = create_access_token({"sub": str(user.id)})
    resp = RedirectResponse("/", status_code=303)
    resp.set_cookie("access_token", token, httponly=True, samesite="lax", max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    return resp


@app.get("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie("access_token")
    return resp


@app.get("/blogs", response_class=HTMLResponse)
def blogs_page(request: Request, db: Session = Depends(db_dep), _: User = Depends(get_current_user)):
    blogs = db.query(Blog).all()
    return templates.TemplateResponse("blogs.html", {"request": request, "blogs": blogs})


@app.post("/blogs")
def create_blog(
    request: Request,
    name: str = Form(...), wp_url: str = Form(...), wp_user: str = Form(...), wp_app_password: str = Form(...),
    niche: str = Form(""), default_prompt: str = Form(""), is_active: bool = Form(False),
    articles_per_week: int = Form(1), preferred_days: str = Form("0,2,4"), preferred_time: str = Form("09:00"),
    db: Session = Depends(db_dep), _: User = Depends(get_current_user)
):
    blog = Blog(name=name, wp_url=wp_url, wp_user=wp_user, wp_app_password_encrypted=encrypt(wp_app_password), niche=niche,
                default_prompt=default_prompt, is_active=is_active, articles_per_week=articles_per_week,
                preferred_days=preferred_days, preferred_time=preferred_time)
    db.add(blog)
    db.commit()
    return RedirectResponse("/blogs", status_code=303)


@app.post("/blogs/{blog_id}/delete")
def delete_blog(blog_id: int, db: Session = Depends(db_dep), _: User = Depends(get_current_user)):
    blog = db.get(Blog, blog_id)
    if blog:
        db.delete(blog)
        db.commit()
    return RedirectResponse("/blogs", status_code=303)


@app.post("/blogs/test-connection")
async def test_connection(wp_url: str = Form(...), wp_user: str = Form(...), wp_app_password: str = Form(...), _: User = Depends(get_current_user)):
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{wp_url.rstrip('/')}/wp-json/wp/v2/users/me", auth=httpx.BasicAuth(wp_user, wp_app_password))
            r.raise_for_status()
            return {"ok": True, "user": r.json().get("name", "ok")}
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})


@app.get("/prompts", response_class=HTMLResponse)
def prompts_page(request: Request, db: Session = Depends(db_dep), _: User = Depends(get_current_user)):
    return templates.TemplateResponse("prompts.html", {"request": request, "prompts": db.query(Prompt).order_by(Prompt.created_at.desc()).all()})


@app.post("/prompts")
def save_prompt(name: str = Form(...), content: str = Form(...), db: Session = Depends(db_dep), _: User = Depends(get_current_user)):
    db.add(Prompt(name=name, content=content))
    db.commit()
    return RedirectResponse("/prompts", status_code=303)


@app.post("/prompts/{prompt_id}/delete")
def delete_prompt(prompt_id: int, db: Session = Depends(db_dep), _: User = Depends(get_current_user)):
    p = db.get(Prompt, prompt_id)
    if p:
        db.delete(p)
        db.commit()
    return RedirectResponse("/prompts", status_code=303)


@app.get("/articles/new", response_class=HTMLResponse)
def article_form(request: Request, db: Session = Depends(db_dep), _: User = Depends(get_current_user)):
    return templates.TemplateResponse("articles_new.html", {"request": request, "blogs": db.query(Blog).filter(Blog.is_active.is_(True)).all(), "prompts": db.query(Prompt).all()})


@app.get("/blogs/{blog_id}/categories")
async def blog_categories(blog_id: int, db: Session = Depends(db_dep), _: User = Depends(get_current_user)):
    blog = db.get(Blog, blog_id)
    if not blog:
        raise HTTPException(status_code=404)
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{blog.wp_url.rstrip('/')}/wp-json/wp/v2/categories", auth=wp_headers(blog))
            r.raise_for_status()
            return [{"id": c["id"], "name": c["name"]} for c in r.json()]
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


def wc(text: str):
    return len([w for w in text.split() if w.strip()])


@app.post("/articles/generate")
async def generate_article(
    keyword: str = Form(...), prompt_text: str = Form(""), prompt_id: int = Form(0), title: str = Form(""), word_target: int = Form(1200),
    ai_model: str = Form("gpt-4o"), blog_id: int = Form(...), category_name: str = Form(""),
    affiliates_enabled: bool = Form(False), affiliate_products: str = Form(""),
    image_enabled: bool = Form(False), image_mode: str = Form("auto"), image_prompt: str = Form(""),
    batch_enabled: bool = Form(False), batch_keywords: str = Form(""),
    db: Session = Depends(db_dep), _: User = Depends(get_current_user)
):
    settings = ensure_settings(db)
    openai_key = decrypt(settings.openai_key_encrypted)
    gemini_key = decrypt(settings.gemini_key_encrypted)
    blog = db.get(Blog, blog_id)
    if not blog:
        raise HTTPException(status_code=404, detail="Blog não encontrado")

    base_prompt = prompt_text
    if prompt_id:
        p = db.get(Prompt, prompt_id)
        if p:
            base_prompt = p.content

    keywords = [keyword]
    if batch_enabled and batch_keywords.strip():
        keywords = [k.strip() for k in batch_keywords.splitlines() if k.strip()]

    created_ids = []
    for kw in keywords:
        work_title = title or f"Guia completo sobre {kw}"
        instruction = f"{base_prompt}\nEscreva artigo SEO em português sobre {kw}. Título: {work_title}. Meta de {word_target} palavras."
        if affiliates_enabled and affiliate_products:
            instruction += f" Insira naturalmente os links: {affiliate_products}."
        content = await generate_text(ai_model, instruction, openai_key, gemini_key)
        attempts = 0
        while attempts < 3:
            actual = wc(content)
            if word_target * 0.9 <= actual <= word_target * 1.1:
                break
            adjust = "expanda" if actual < word_target else "reduza"
            content = await generate_text(ai_model, f"{adjust} o texto abaixo para cerca de {word_target} palavras:\n{content}", openai_key, gemini_key)
            attempts += 1
        image_url = None
        if image_enabled:
            image_url = "https://placehold.co/1024x1024?text=Capa"
        article = Article(title=work_title, keyword=kw, content=content, ai_model=ai_model, word_count=wc(content),
                          status="Aguardando revisão", category_name=category_name, blog_id=blog.id, image_url=image_url)
        db.add(article)
        db.flush()
        created_ids.append(article.id)
    db.commit()
    return {"created_ids": created_ids}


@app.get("/articles/review", response_class=HTMLResponse)
def review_page(request: Request, db: Session = Depends(db_dep), _: User = Depends(get_current_user)):
    articles = db.query(Article).filter(Article.status == "Aguardando revisão").order_by(Article.created_at.desc()).all()
    return templates.TemplateResponse("review.html", {"request": request, "articles": articles})


@app.post("/articles/{article_id}/approve")
async def approve_article(article_id: int, action: str = Form("draft"), schedule_at: str = Form(""), db: Session = Depends(db_dep), _: User = Depends(get_current_user)):
    article = db.get(Article, article_id)
    if not article:
        raise HTTPException(status_code=404)
    blog = db.get(Blog, article.blog_id)
    status_post = "draft"
    date_gmt = None
    if action == "publish":
        status_post = "publish"
        article.status = "Publicado"
        article.published_at = datetime.utcnow()
    elif action == "schedule" and schedule_at:
        status_post = "future"
        dt = datetime.fromisoformat(schedule_at)
        date_gmt = dt.strftime("%Y-%m-%dT%H:%M:%S")
        article.status = "Agendado"
        article.scheduled_for = dt
    elif action == "smart":
        dt = smart_slots(blog, 1, db)[0]
        status_post = "future"
        date_gmt = dt.strftime("%Y-%m-%dT%H:%M:%S")
        article.status = "Agendado"
        article.scheduled_for = dt
    else:
        article.status = "Rascunho"
    posted = await wp_post(blog, article.title, article.content, None, status_post=status_post, date_gmt=date_gmt)
    article.wp_post_id = str(posted.get("id"))
    db.commit()
    return RedirectResponse("/articles/review", status_code=303)


@app.post("/articles/{article_id}/regenerate")
async def regenerate_article(article_id: int, db: Session = Depends(db_dep), _: User = Depends(get_current_user)):
    article = db.get(Article, article_id)
    settings = ensure_settings(db)
    new_content = await generate_text(article.ai_model, f"Reescreva do zero sobre {article.keyword} com SEO melhor", decrypt(settings.openai_key_encrypted), decrypt(settings.gemini_key_encrypted))
    article.content = new_content
    article.word_count = wc(new_content)
    db.commit()
    return RedirectResponse("/articles/review", status_code=303)


@app.post("/articles/{article_id}/delete")
def remove_article(article_id: int, db: Session = Depends(db_dep), _: User = Depends(get_current_user)):
    article = db.get(Article, article_id)
    if article:
        db.delete(article)
        db.commit()
    return RedirectResponse("/articles/review", status_code=303)


@app.get("/articles/history", response_class=HTMLResponse)
def history_page(request: Request, blog_id: int = 0, status_filter: str = "", q: str = "", db: Session = Depends(db_dep), _: User = Depends(get_current_user)):
    query = db.query(Article)
    if blog_id:
        query = query.filter(Article.blog_id == blog_id)
    if status_filter:
        query = query.filter(Article.status == status_filter)
    if q:
        query = query.filter(Article.title.ilike(f"%{q}%"))
    articles = query.order_by(Article.created_at.desc()).all()
    return templates.TemplateResponse("history.html", {"request": request, "articles": articles, "blogs": db.query(Blog).all()})


@app.get("/articles/export")
def export_articles(fmt: str = "csv", db: Session = Depends(db_dep), _: User = Depends(get_current_user)):
    rows = []
    for a in db.query(Article).all():
        rows.append({"titulo": a.title, "blog": a.blog.name if a.blog else "", "categoria": a.category_name,
                     "modelo_ia": a.ai_model, "palavras": a.word_count, "status": a.status,
                     "data_criacao": a.created_at, "data_publicacao": a.published_at})
    if fmt == "xlsx":
        df = pd.DataFrame(rows)
        out = io.BytesIO()
        df.to_excel(out, index=False)
        out.seek(0)
        return StreamingResponse(out, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": "attachment; filename=historico.xlsx"})
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()) if rows else ["titulo"])
    writer.writeheader()
    writer.writerows(rows)
    return Response(content=out.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=historico.csv"})


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, db: Session = Depends(db_dep), _: User = Depends(get_current_user)):
    settings = ensure_settings(db)
    return templates.TemplateResponse("settings.html", {"request": request, "settings": settings})


@app.post("/settings")
def save_settings(openai_key: str = Form(""), gemini_key: str = Form(""), status_interval_minutes: int = Form(30),
                  new_username: str = Form(""), new_password: str = Form(""),
                  db: Session = Depends(db_dep), user: User = Depends(get_current_user)):
    settings = ensure_settings(db)
    if openai_key:
        settings.openai_key_encrypted = encrypt(openai_key)
    if gemini_key:
        settings.gemini_key_encrypted = encrypt(gemini_key)
    settings.status_interval_minutes = status_interval_minutes
    if new_username:
        user.username = new_username
    if new_password:
        user.password_hash = pwd_context.hash(new_password)
    db.commit()
    if scheduler.get_job("status_refresh"):
        scheduler.reschedule_job("status_refresh", trigger="interval", minutes=status_interval_minutes)
    return RedirectResponse("/settings", status_code=303)
