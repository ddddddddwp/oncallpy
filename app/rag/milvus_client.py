"""Milvus 客户端工厂模块"""

# 该管理器负责 collection 生命周期；领域读写仍通过 MilvusVectorStore 适配器完成。
from loguru import logger
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    MilvusException,
    connections,
    utility,
)

from app.config import config


class MilvusClientManager:
    """Milvus 客户端管理器"""

    # 常量定义
    # schema 常量必须与 embedding provider 和 MilvusVectorStore 的字段映射保持一致。
    VECTOR_DIM: int = 1024  # 统一使用 1024 维
    ID_MAX_LENGTH: int = 100
    CONTENT_MAX_LENGTH: int = 8000
    DEFAULT_SHARD_NUMBER: int = 2

    def __init__(self) -> None:
        """初始化 Milvus 客户端管理器"""
        self._client: MilvusClient | None = None
        self._collection: Collection | None = None
        self.collection_name = config.milvus_collection_name

    def connect(self) -> MilvusClient:
        """
        连接到 Milvus 服务器并初始化 collection

        Returns:
            MilvusClient: Milvus 客户端实例

        Raises:
            RuntimeError: 连接或初始化失败时抛出
        """
        try:
            logger.info(f"正在连接到 Milvus: {config.milvus_host}:{config.milvus_port}")

            # 建立连接
            # 旧 Collection API 依赖全局 alias，MilvusClient 则提供较新的实例式 API；两者都需初始化。
            connections.connect(
                alias="default",
                host=config.milvus_host,
                port=str(config.milvus_port),
                db_name=config.milvus_database_name,
                timeout=config.milvus_timeout / 1000,  # 转换为秒
            )

            # 创建客户端
            uri = f"http://{config.milvus_host}:{config.milvus_port}"
            self._client = MilvusClient(
                uri=uri,
                db_name=config.milvus_database_name,
            )

            logger.info("成功连接到 Milvus")

            # 检查并创建 collection
            # 首次运行创建 schema；已有 collection 则先验证关键的向量维度契约。
            if not self._collection_exists():
                logger.info(f"collection '{self.collection_name}' 不存在，正在创建...")
                self._create_collection()
                logger.info(f"成功创建 collection '{self.collection_name}'")
            else:
                logger.info(f"collection '{self.collection_name}' 已存在")
                self._collection = Collection(self.collection_name)

                # 检查向量维度是否匹配
                schema = self._collection.schema
                vector_field = None
                existing_dim = None
                # 只检查 vector 字段，不对其他字段做隐式迁移，避免意外破坏线上数据。
                for field in schema.fields:
                    if field.name == "vector":
                        vector_field = field
                        break

                if vector_field and hasattr(vector_field, 'params') and 'dim' in vector_field.params:
                    existing_dim = vector_field.params['dim']
                    # 维度不匹配无法继续相似度计算，现有策略是明确告警后重建 collection。
                    if existing_dim != self.VECTOR_DIM:
                        logger.warning(
                            f"检测到向量维度不匹配！当前 collection 维度: {existing_dim}, 配置维度: {self.VECTOR_DIM}"
                        )
                        logger.info(f"正在删除旧 collection '{self.collection_name}'...")
                        _ = utility.drop_collection(self.collection_name)
                        logger.info(f"正在重新创建 collection '{self.collection_name}'...")
                        self._create_collection()
                        logger.info(f"成功重新创建 collection，维度: {self.VECTOR_DIM}")
                    else:
                        logger.info(f"向量维度匹配: {self.VECTOR_DIM}")

            # 加载 collection
            # collection 必须 load 到查询节点内存后，后续检索和删除才可稳定执行。
            self._load_collection()

            return self._client

        # 所有失败路径都先 close 已建立的部分资源，再统一转换为调用方可处理的 RuntimeError。
        except MilvusException as e:
            logger.error(f"Milvus 操作失败: {e}")
            self.close()
            raise RuntimeError(f"Milvus 操作失败: {e}") from e
        except ConnectionError as e:
            logger.error(f"连接 Milvus 失败: {e}")
            self.close()
            raise RuntimeError(f"连接 Milvus 失败: {e}") from e
        except Exception as e:
            logger.error(f"连接 Milvus 失败: {e}")
            self.close()
            raise RuntimeError(f"连接 Milvus 失败: {e}") from e

    def _collection_exists(self) -> bool:
        """检查 collection 是否存在"""
        # pymilvus 的类型标注可能不准确，实际返回 bool
        result = utility.has_collection(self.collection_name)
        return bool(result)  # type: ignore[arg-type]

    def _create_collection(self) -> None:
        """创建 biz collection"""
        # 定义字段
        # 禁用 dynamic field，所有业务 metadata 收纳进 JSON 字段，schema 保持小而稳定。
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=self.ID_MAX_LENGTH,
                is_primary=True,
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.VECTOR_DIM,
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=self.CONTENT_MAX_LENGTH,
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
            ),
        ]

        # 创建 schema
        schema = CollectionSchema(
            fields=fields,
            description="Business knowledge collection",
            enable_dynamic_field=False,
        )

        # 创建 collection
        # shard 数决定写入分片并行度；这里使用固定默认值保持部署行为可预测。
        self._collection = Collection(
            name=self.collection_name,
            schema=schema,
            num_shards=self.DEFAULT_SHARD_NUMBER,
        )

        # 创建索引
        self._create_index()

    def _create_index(self) -> None:
        """为 vector 字段创建索引"""
        if self._collection is None:
            raise RuntimeError("Collection 未初始化")

        # index_type/nlist 控制构建方式，metric_type 必须与查询时的 search params 对齐。
        index_params = {
            "metric_type": config.rag_vector_metric_type,
            "index_type": config.rag_vector_index_type,
            "params": {"nlist": config.rag_vector_nlist},
        }

        _ = self._collection.create_index(
            field_name="vector",
            index_params=index_params,
        )

        logger.info("成功为 vector 字段创建索引")

    def _load_collection(self) -> None:
        """加载 collection 到内存"""
        if self._collection is None:
            self._collection = Collection(self.collection_name)

        # 检查 collection 是否已加载（兼容多版本）
        # pymilvus 不同版本暴露的加载状态 API 不一致，先尝试新版再降级到异常判断。
        try:
            # 方法 1: 尝试使用 utility.load_state（新版本）
            load_state = utility.load_state(self.collection_name)
            # load_state 返回字符串或枚举，如 "Loaded" 或 "NotLoad"
            state_name = getattr(load_state, "name", str(load_state))
            if state_name != "Loaded":
                self._collection.load()
                logger.info(f"成功加载 collection '{self.collection_name}'")
            else:
                logger.info(f"Collection '{self.collection_name}' 已加载")
        except AttributeError:
            # 旧版本没有 utility.load_state，直接 load 并把“已加载”视为幂等成功。
            # 方法 2: 直接尝试加载，捕获 "already loaded" 异常
            try:
                self._collection.load()
                logger.info(f"成功加载 collection '{self.collection_name}'")
            except MilvusException as e:
                error_msg = str(e).lower()
                if "already loaded" in error_msg or "loaded" in error_msg:
                    logger.info(f"Collection '{self.collection_name}' 已加载")
                else:
                    raise
        except Exception as e:
            logger.error(f"加载 collection 失败: {e}")
            raise

    def get_collection(self) -> Collection:
        """
        获取 collection 实例

        Returns:
            Collection: collection 实例

        Raises:
            RuntimeError: collection 未初始化时抛出
        """
        if self._collection is None:
            raise RuntimeError("Collection 未初始化，请先调用 connect()")
        return self._collection

    def health_check(self) -> bool:
        """
        健康检查

        Returns:
            bool: True 表示健康，False 表示异常
        """
        try:
            # 本地 client 状态先做快速判断，再通过连接注册表验证底层连接仍存在。
            if self._client is None:
                return False

            # 尝试列出 connections
            _ = connections.list_connections()
            return True

        except (MilvusException, ConnectionError) as exc:
            logger.error("Milvus health check failed: {}", type(exc).__name__)
            return False
        except Exception as exc:
            logger.error("Milvus health check failed: {}", type(exc).__name__)
            return False

    def close(self) -> None:
        """关闭连接"""
        # release collection 与 disconnect connection 独立尝试，单一步骤失败不阻止另一资源释放。
        errors = []

        try:
            if self._collection is not None:
                self._collection.release()
                self._collection = None
        except Exception as e:
            errors.append(f"释放 collection 失败: {e}")

        try:
            if connections.has_connection("default"):
                connections.disconnect("default")
        except Exception as e:
            errors.append(f"断开连接失败: {e}")

        self._client = None

        if errors:
            error_msg = "; ".join(errors)
            logger.error(f"关闭 Milvus 连接时出现错误: {error_msg}")
        else:
            logger.info("已关闭 Milvus 连接")

    def __enter__(self) -> "MilvusClientManager":
        """上下文管理器入口"""
        # 上下文入口保证 ready，出口无论业务是否抛错都会调用 close。
        _ = self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object
    ) -> None:
        """上下文管理器退出"""
        self.close()


# 全局单例
# 全局只保存管理器对象；网络连接仍由显式 connect 或应用生命周期触发。
milvus_manager = MilvusClientManager()
