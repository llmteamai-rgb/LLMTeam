"""
PostgreSQL tenant store implementation.

This store persists tenant configurations in PostgreSQL and is suitable
for production deployments.
"""

import json
from datetime import datetime
from typing import Any, List, Optional

from llmteam.tenancy.models import TenantConfig, TenantTier


class PostgresTenantStore:
    """
    PostgreSQL implementation of TenantStore.
    
    Stores tenant configurations in a PostgreSQL table. Suitable for:
    - Production deployments
    - Multi-instance deployments
    - Persistent storage requirements
    
    Required table schema:
    ```sql
    CREATE TABLE tenants (
        tenant_id VARCHAR(255) PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        tier VARCHAR(50) NOT NULL,
        config JSONB NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
        is_active BOOLEAN NOT NULL DEFAULT TRUE
    );
    
    CREATE INDEX idx_tenants_tier ON tenants(tier);
    CREATE INDEX idx_tenants_active ON tenants(is_active);
    ```
    
    Example:
        import asyncpg
        
        pool = await asyncpg.create_pool(dsn)
        store = PostgresTenantStore(pool)
        manager = TenantManager(store)
    """
    
    def __init__(
        self, 
        pool: Any,  # asyncpg.Pool
        table_name: str = "tenants",
    ):
        """
        Initialize PostgresTenantStore.
        
        Args:
            pool: asyncpg connection pool
            table_name: Name of the tenants table
        """
        self.pool = pool
        self.table_name = table_name
    
    async def get(self, tenant_id: str) -> Optional[TenantConfig]:
        """
        Get a tenant by ID.
        
        Args:
            tenant_id: The tenant ID to look up
            
        Returns:
            TenantConfig if found, None otherwise
        """
        query = f"""
            SELECT tenant_id, name, tier, config, created_at, updated_at, is_active
            FROM {self.table_name}
            WHERE tenant_id = $1
        """
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, tenant_id)
        
        if row is None:
            return None
        
        return self._row_to_config(row)
    
    async def create(self, config: TenantConfig) -> None:
        """
        Create a new tenant.
        
        Args:
            config: Tenant configuration to store
        """
        query = f"""
            INSERT INTO {self.table_name} (
                tenant_id, name, tier, config, created_at, updated_at, is_active
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        """
        
        config_json = self._config_to_json(config)
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                config.tenant_id,
                config.name,
                config.tier.value,
                config_json,
                config.created_at,
                config.updated_at,
                config.is_active,
            )
    
    async def update(self, config: TenantConfig) -> None:
        """
        Update an existing tenant.
        
        Args:
            config: Updated tenant configuration
        """
        query = f"""
            UPDATE {self.table_name}
            SET name = $2, tier = $3, config = $4, updated_at = $5, is_active = $6
            WHERE tenant_id = $1
        """
        
        config_json = self._config_to_json(config)
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                config.tenant_id,
                config.name,
                config.tier.value,
                config_json,
                config.updated_at,
                config.is_active,
            )
    
    async def delete(self, tenant_id: str) -> None:
        """
        Delete a tenant (soft delete by setting is_active = false).
        
        Args:
            tenant_id: The tenant ID to delete
        """
        query = f"""
            UPDATE {self.table_name}
            SET is_active = FALSE, updated_at = $2
            WHERE tenant_id = $1
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(query, tenant_id, datetime.now())
    
    async def hard_delete(self, tenant_id: str) -> None:
        """
        Permanently delete a tenant.
        
        Args:
            tenant_id: The tenant ID to delete
        """
        query = f"DELETE FROM {self.table_name} WHERE tenant_id = $1"
        
        async with self.pool.acquire() as conn:
            await conn.execute(query, tenant_id)
    
    async def list(
        self, 
        limit: int = 100, 
        offset: int = 0,
        include_inactive: bool = False,
    ) -> List[TenantConfig]:
        """
        List tenants with pagination.
        
        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            include_inactive: Whether to include inactive tenants
            
        Returns:
            List of TenantConfig objects
        """
        if include_inactive:
            query = f"""
                SELECT tenant_id, name, tier, config, created_at, updated_at, is_active
                FROM {self.table_name}
                ORDER BY created_at DESC
                LIMIT $1 OFFSET $2
            """
            params = [limit, offset]
        else:
            query = f"""
                SELECT tenant_id, name, tier, config, created_at, updated_at, is_active
                FROM {self.table_name}
                WHERE is_active = TRUE
                ORDER BY created_at DESC
                LIMIT $1 OFFSET $2
            """
            params = [limit, offset]
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        
        return [self._row_to_config(row) for row in rows]
    
    async def count(self, include_inactive: bool = False) -> int:
        """
        Get total number of tenants.
        
        Args:
            include_inactive: Whether to count inactive tenants
            
        Returns:
            Number of tenants
        """
        if include_inactive:
            query = f"SELECT COUNT(*) FROM {self.table_name}"
        else:
            query = f"SELECT COUNT(*) FROM {self.table_name} WHERE is_active = TRUE"
        
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query)
    
    async def list_by_tier(self, tier: TenantTier) -> List[TenantConfig]:
        """
        List all tenants of a specific tier.
        
        Args:
            tier: The tier to filter by
            
        Returns:
            List of TenantConfig objects
        """
        query = f"""
            SELECT tenant_id, name, tier, config, created_at, updated_at, is_active
            FROM {self.table_name}
            WHERE tier = $1 AND is_active = TRUE
            ORDER BY created_at DESC
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, tier.value)
        
        return [self._row_to_config(row) for row in rows]
    
    def _row_to_config(self, row: Any) -> TenantConfig:
        """Convert a database row to TenantConfig."""
        config_data = row["config"]
        if isinstance(config_data, str):
            config_data = json.loads(config_data)
        
        return TenantConfig(
            tenant_id=row["tenant_id"],
            name=row["name"],
            tier=TenantTier(row["tier"]),
            max_concurrent_pipelines=config_data.get("max_concurrent_pipelines"),
            max_agents_per_pipeline=config_data.get("max_agents_per_pipeline"),
            max_requests_per_minute=config_data.get("max_requests_per_minute"),
            features_enabled=set(config_data.get("features_enabled", [])),
            features_disabled=set(config_data.get("features_disabled", [])),
            allowed_actions=set(config_data.get("allowed_actions", [])),
            blocked_actions=set(config_data.get("blocked_actions", [])),
            data_region=config_data.get("data_region", "default"),
            encryption_key_id=config_data.get("encryption_key_id", ""),
            audit_retention_days=config_data.get("audit_retention_days", 90),
            metadata=config_data.get("metadata", {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            is_active=row["is_active"],
        )
    
    def _config_to_json(self, config: TenantConfig) -> str:
        """Convert TenantConfig to JSON for storage."""
        data = {
            "max_concurrent_pipelines": config.max_concurrent_pipelines,
            "max_agents_per_pipeline": config.max_agents_per_pipeline,
            "max_requests_per_minute": config.max_requests_per_minute,
            "features_enabled": list(config.features_enabled),
            "features_disabled": list(config.features_disabled),
            "allowed_actions": list(config.allowed_actions),
            "blocked_actions": list(config.blocked_actions),
            "data_region": config.data_region,
            "encryption_key_id": config.encryption_key_id,
            "audit_retention_days": config.audit_retention_days,
            "metadata": config.metadata,
        }
        return json.dumps(data)


# SQL schema for reference
POSTGRES_SCHEMA = """
-- Tenants table
CREATE TABLE IF NOT EXISTS tenants (
    tenant_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    tier VARCHAR(50) NOT NULL DEFAULT 'free',
    config JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_tenants_tier ON tenants(tier);
CREATE INDEX IF NOT EXISTS idx_tenants_active ON tenants(is_active);
CREATE INDEX IF NOT EXISTS idx_tenants_created ON tenants(created_at);

-- Updated at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_tenants_updated_at
    BEFORE UPDATE ON tenants
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
"""
