from sqlalchemy import Column, Table, Uuid

from app.database.base import Base

# Supabase-owned table; stub exists so SQLAlchemy can resolve profiles.id FK metadata.
auth_users = Table(
    "users",
    Base.metadata,
    Column("id", Uuid, primary_key=True),
    schema="auth",
    info={"skip_autogenerate": True},
)
