-- v1.0.16: 多候选并行模拟赛马（Champion-Challenger Multi-Account Simulation Arena）
-- 1. 放开 simulation_accounts 单用户唯一约束，新增 name/role/bound_model_id/bound_strategy_id/is_default/is_archived 列
-- 2. 存量账户回填 is_default = TRUE，保持与历史单账户链路向后兼容
-- 幂等可重放

DO $$
BEGIN
    IF to_regclass('public.simulation_accounts') IS NOT NULL THEN
        ALTER TABLE simulation_accounts
            ADD COLUMN IF NOT EXISTS name VARCHAR(128) NOT NULL DEFAULT '默认模拟账户';
        ALTER TABLE simulation_accounts
            ADD COLUMN IF NOT EXISTS role VARCHAR(32) NOT NULL DEFAULT 'champion';
        ALTER TABLE simulation_accounts
            ADD COLUMN IF NOT EXISTS bound_model_id VARCHAR(128);
        ALTER TABLE simulation_accounts
            ADD COLUMN IF NOT EXISTS bound_strategy_id VARCHAR(96);
        ALTER TABLE simulation_accounts
            ADD COLUMN IF NOT EXISTS is_default BOOLEAN NOT NULL DEFAULT FALSE;
        ALTER TABLE simulation_accounts
            ADD COLUMN IF NOT EXISTS is_archived BOOLEAN NOT NULL DEFAULT FALSE;

        -- 存量单账户（account_id 形如 sim:{tenant}:{user}）标记为默认账户
        UPDATE simulation_accounts
           SET is_default = TRUE
         WHERE account_id = 'sim:' || tenant_id || ':' || user_id
           AND is_default = FALSE;

        -- 将原 UNIQUE 索引降级为普通索引，允许单用户持有多个并行模拟账户
        DROP INDEX IF EXISTS idx_simulation_accounts_tenant_user;
        CREATE INDEX IF NOT EXISTS idx_simulation_accounts_tenant_user
            ON simulation_accounts (tenant_id, user_id);
        CREATE INDEX IF NOT EXISTS idx_simulation_accounts_model_active
            ON simulation_accounts (tenant_id, bound_model_id, is_archived);
    END IF;
END $$;
