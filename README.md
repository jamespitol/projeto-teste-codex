# Blog Manager

Sistema web para gerenciamento de múltiplos blogs WordPress com geração de artigos por IA.

## Stack
- FastAPI + SQLAlchemy + SQLite
- APScheduler
- HTML/CSS/JS
- Criptografia Fernet (`cryptography`)

## Instalação
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuração
1. Ajuste o arquivo `.env`.
2. Execute a aplicação:
```bash
uvicorn main:app --reload
```
3. Abra `http://127.0.0.1:8000`.
4. No primeiro acesso, use `/setup` para criar usuário/senha.

## Segurança
- Na primeira execução, o sistema gera automaticamente `.key` com chave mestra Fernet.
- Credenciais de blogs e APIs são armazenadas criptografadas no banco SQLite.
- Sessão com JWT em cookie HTTP-only, com expiração de 1 hora.

## Módulos implementados
- Dashboard
- Cadastro de blogs + teste de conexão WordPress
- Geração de artigos (inclui lote, ajuste de tamanho e revisão)
- Agendamento inteligente por frequência/dias/horário
- Histórico com filtros e exportação CSV/XLSX
- Prompts salvos
- Configurações (chaves API, intervalo de sincronização, usuário/senha)

## Observações
- Integrações com OpenAI/Gemini/WordPress possuem tratamento de erros.
- O job do APScheduler verifica status de artigos periodicamente.
