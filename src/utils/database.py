#/core/database.py
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base
from src.utils.config_loader import settings

#创建数据库核心

#1.连接数据库引擎
#1.1 定义连接URL

DATABASE_URL = settings.get_database_url()

#1.2 创建数据库引擎
engine = create_engine(
    url = DATABASE_URL,
    #连接池大小
    pool_size = settings.DATABASE_POOL_SIZE,
    #连接池最大溢出数
    max_overflow = settings.DATABASE_MAX_OVERFLOW,
    # ★ 连接回收时间与「获取连接超时」是两回事，原先误把 RECYCLE(3600s) 当成了超时，
    #   后果是拿不到连接时会挂满一小时才报错。
    pool_recycle = settings.DATABASE_POOL_RECYCLE,
    pool_timeout = settings.DATABASE_POOL_TIMEOUT,

    # ★ 生产不许把每条 SQL 打到 stdout（原先写死 True）
    echo = settings.DEBUG,
)


#2.创建会话工厂
session_creater = sessionmaker(
    bind = engine,
    autocommit = False,
    autoflush = False,
    )


#3.定义一个创建会话的函数/方法
@contextmanager
def get_db_session():
    session = session_creater()
    try:
        yield session
        session.commit()      # 统一提交
    except Exception:           
        session.rollback()      # 统一回滚
        raise
    finally:
        session.close()         # 统一关闭


#4.获取数据库元数据
Base = declarative_base()