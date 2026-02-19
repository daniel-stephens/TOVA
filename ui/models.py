# models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.dialects.postgresql import JSON

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.String(36), primary_key=True)  # uuid4 string
    name = db.Column(db.String(255))
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=True)
    auth_source = db.Column(db.String(50))
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password, method="pbkdf2:sha256")

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class UserConfig(db.Model):
    """
    Stores a per-user configuration blob (JSON) along with timestamps.
    """

    __tablename__ = "user_configs"

    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), primary_key=True)
    config = db.Column(JSON, nullable=False, default=dict)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user = db.relationship("User", backref=db.backref("config", uselist=False, cascade="all, delete-orphan"))


class AuditLog(db.Model):
    """Audit log for admin visibility: logins, user/model/corpus actions, admin actions."""
    __tablename__ = "audit_logs"

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    actor_id = db.Column(db.String(36), nullable=True)  # user who performed action (None for system)
    action = db.Column(db.String(64), nullable=False)  # e.g. login, user_created, user_deleted, corpus_deleted
    target_type = db.Column(db.String(32), nullable=True)  # user, corpus, model
    target_id = db.Column(db.String(255), nullable=True)
    details = db.Column(db.String(1024), nullable=True)  # optional JSON or short description


class ChatMessage(db.Model):
    """Stores chat messages per user and model for the dashboard assistant."""
    __tablename__ = "chat_messages"

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, index=True)
    model_id = db.Column(db.String(255), nullable=False, index=True)  # topic model id messages belong to
    role = db.Column(db.String(16), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (db.Index("ix_chat_messages_user_model", "user_id", "model_id"),)

    user = db.relationship("User", backref=db.backref("chat_messages", lazy="dynamic", cascade="all, delete-orphan"))