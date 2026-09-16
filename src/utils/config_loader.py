# src/utils/config_loader.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """向上查找项目根（即含 .env 的目录）。

    ★ 为什么不写死 Path(__file__).parent.parent：
      本文件在目录重构中从 core/config.py 挪到了 src/utils/config_loader.py，
      层级从 2 层变成 3 层。写死的层级会**静默**指到 src/ 而不是项目根 ——
      而 .env 找不到时 pydantic-settings 会回落到默认值
      （localhost / 3306 / scada_db），**照样连得上**。
      于是你不会看到任何报错，直到某天库名或密码变了才发现配置根本没生效。
      向上查找可以让文件随便挪，永远找得到。
    """
    here = (start or Path(__file__)).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / ".env").is_file():
            return parent
    # 兜底：没有 .env 时退回三级上级（src/utils/x.py → 项目根）
    return here.parents[2]


#工程路径
PROJECT_PATH = find_project_root()
#环境路径
ENV_PATH = PROJECT_PATH / '.env'

class Setting(BaseSettings):
    #1、数据库配置
    DATABASE_TYPE : str = "mysql"
    DATABASE_HOST : str = "localhost"
    DATABASE_PORT : int = 3306
    DATABASE_NAME : str 
    DATABASE_USER : str
    DATABASE_PASSWORD : str 


    DATABASE_POOL_SIZE: int
    DATABASE_MAX_OVERFLOW: int
    DATABASE_POOL_RECYCLE: int
    # ★ 必须在此声明：.env.example 里已经有这个键，而 pydantic-settings 默认
    #   extra='forbid'，未声明的键会让新人照模板建完 .env 后直接 ValidationError。
    DATABASE_POOL_TIMEOUT: int = 30

    # 调试开关：为 True 时 SQLAlchemy 会把每条 SQL 打到 stdout
    DEBUG: bool = False


    #本地开发环境加载.env环境配置信息
    model_config = SettingsConfigDict(env_file=ENV_PATH, env_file_encoding='utf-8')
    #线上环境加载环境变量直接通过os.environ获取系统环境变量，保证配置信息安全
    #数据库连接URL
    def get_database_url(self):
        return f"{self.DATABASE_TYPE}+pymysql://{self.DATABASE_USER}:{self.DATABASE_PASSWORD}@{self.DATABASE_HOST}:{self.DATABASE_PORT}/{self.DATABASE_NAME}?charset=utf8"

@lru_cache(maxsize=1)
def get_settings():
    # 获取配置信息
    return Setting()

#全局单例
settings = get_settings()