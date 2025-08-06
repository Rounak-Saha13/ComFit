# Comfit Copilot Backend

This is the backend API for the Comfit Copilot chat application, built with FastAPI and integrated with Supabase for authentication and PostgreSQL for data storage.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the backend directory with the following variables:

```env
# database
DATABASE_URL=postgresql+asyncpg://postgres:5PyYhS3TXZ2NxYcV@db.tyswhhteurchuzkngqja.supabase.co:5432/postgres

# ollama creds - 4070
OLLAMA_BASE_URL=http://192.168.0.240:11434

# supabase creds
SUPABASE_URL=
SUPABASE_ANON_KEY=
NEXT_PUBLIC_SUPABASE_ROLE=

# google api
GOOGLE_API_KEY=
GOOGLE_CSE_ID=
```

### 3. Start the Server

```bash
python main.py
```

The server will start on `http://localhost:8000` by default.
