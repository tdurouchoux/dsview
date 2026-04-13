-- PostgreSQL script to create read-only user for dsview
-- Password is read from environment variable READONLY_USER_PASSWORD

DO $$
BEGIN
    -- Create the read-only user with password from environment variable
    EXECUTE format('CREATE USER dsview_ro WITH PASSWORD %L', current_setting('app.readonly_user_password', true));
    EXCEPTION WHEN SQLSTATE '42710' THEN
        RAISE NOTICE 'User dsview_ro already exists, skipping creation';
END $$;

-- Grant database connection
GRANT CONNECT ON DATABASE defaultdb TO dsview_ro;

-- Grant usage on all three schemas
GRANT USAGE ON SCHEMA content TO dsview_ro;
GRANT USAGE ON SCHEMA extraction TO dsview_ro;
GRANT USAGE ON SCHEMA labels TO dsview_ro;

-- Grant SELECT on all tables in each schema
GRANT SELECT ON ALL TABLES IN SCHEMA content TO dsview_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA extraction TO dsview_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA labels TO dsview_ro;

-- Set default privileges for future tables in each schema
ALTER DEFAULT PRIVILEGES IN SCHEMA content GRANT SELECT ON TABLES TO dsview_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA extraction GRANT SELECT ON TABLES TO dsview_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA labels GRANT SELECT ON TABLES TO dsview_ro;

-- Grant usage on sequences if they exist
GRANT USAGE ON ALL SEQUENCES IN SCHEMA content TO dsview_ro;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA extraction TO dsview_ro;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA labels TO dsview_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA content GRANT USAGE ON SEQUENCES TO dsview_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA extraction GRANT USAGE ON SEQUENCES TO dsview_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA labels GRANT USAGE ON SEQUENCES TO dsview_ro;