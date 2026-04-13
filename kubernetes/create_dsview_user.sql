-- PostgreSQL script to create read-write user for dsview
-- Password is read from environment variable RW_USER_PASSWORD

DO $$
BEGIN
    -- Create the read-write user with password from environment variable
    EXECUTE format('CREATE USER dsview WITH PASSWORD %L', current_setting('app.rw_user_password', true));
    EXCEPTION WHEN SQLSTATE '42710' THEN
        RAISE NOTICE 'User dsview already exists, skipping creation';
END $$;

-- Grant database connection
GRANT CONNECT ON DATABASE defaultdb TO dsview;

-- Grant usage on all three schemas
GRANT USAGE ON SCHEMA content TO dsview;
GRANT USAGE ON SCHEMA extraction TO dsview;
GRANT USAGE ON SCHEMA labels TO dsview;

-- Grant full CRUD on content and extraction schemas
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA content TO dsview;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA extraction TO dsview;

-- Grant only SELECT on labels schema (read-only)
GRANT SELECT ON ALL TABLES IN SCHEMA labels TO dsview;

-- Set default privileges for future tables
-- Full CRUD for content and extraction
ALTER DEFAULT PRIVILEGES IN SCHEMA content GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO dsview;
ALTER DEFAULT PRIVILEGES IN SCHEMA extraction GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO dsview;
-- Read-only for labels
ALTER DEFAULT PRIVILEGES IN SCHEMA labels GRANT SELECT ON TABLES TO dsview;

-- Grant usage and update on sequences for content and extraction
GRANT USAGE, UPDATE ON ALL SEQUENCES IN SCHEMA content TO dsview;
GRANT USAGE, UPDATE ON ALL SEQUENCES IN SCHEMA extraction TO dsview;
ALTER DEFAULT PRIVILEGES IN SCHEMA content GRANT USAGE, UPDATE ON SEQUENCES TO dsview;
ALTER DEFAULT PRIVILEGES IN SCHEMA extraction GRANT USAGE, UPDATE ON SEQUENCES TO dsview;

-- Grant only usage on sequences for labels (read-only)
GRANT USAGE ON ALL SEQUENCES IN SCHEMA labels TO dsview;
ALTER DEFAULT PRIVILEGES IN SCHEMA labels GRANT USAGE ON SEQUENCES TO dsview;
