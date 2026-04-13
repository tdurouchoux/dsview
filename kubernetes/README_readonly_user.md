# PostgreSQL User Setup

This directory contains SQL scripts to create different database users for the dsview application with varying permission levels.

## Available Scripts

1. **`create_readonly_user.sql`** - Creates `dsview_ro` user with read-only access to all schemas
2. **`create_rw_user.sql`** - Creates `dsview_rw` user with read-write access to `content` and `extraction`, read-only to `labels`

## Usage

### For Read-Only User (dsview_ro)

#### 1. Set the password environment variable

```bash
export READONLY_USER_PASSWORD="your_secure_password_here"
```

#### 2. Run the script using psql

```bash
psql -h localhost -U postgres -d defaultdb -v readonly_user_password="$READONLY_USER_PASSWORD" -f create_readonly_user.sql
```

### For Read-Write User (dsview_rw)

#### 1. Set the password environment variable

```bash
export RW_USER_PASSWORD="your_secure_password_here"
```

#### 2. Run the script using psql

```bash
psql -h localhost -U postgres -d defaultdb -v rw_user_password="$RW_USER_PASSWORD" -f create_rw_user.sql
```

### 3. For Kubernetes deployment

Add the read-only password to your Kubernetes secret:

```bash
# Encode the password
READONLY_PASSWORD_BASE64=$(echo -n "$READONLY_USER_PASSWORD" | base64)

# Patch the secret (if it already exists)
kubectl patch secret dsview-secret --type='json' -p="[{\"op\": \"add\", \"path\": \"/data/POSTGRES_READONLY_PASSWORD\", \"value\": \"$READONLY_PASSWORD_BASE64\"}]"
```

Or include it in your secret manifest:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: dsview-secret
type: Opaque
data:
  POSTGRES_PASSWORD: <admin-password-base64>
  POSTGRES_READONLY_PASSWORD: <readonly-password-base64>
  POSTGRES_DSVIEW_PASSWORD: <rw-password-base64>
  # ... other secrets
```

## What the script does

1. Creates user `dsview_ro` with the provided password
2. Grants CONNECT permission on `defaultdb` database
3. Grants USAGE on schemas: `content`, `extraction`, `labels`
4. Grants SELECT on all existing tables in those schemas
5. Sets default privileges so future tables are also readable
6. Grants USAGE on sequences (for auto-increment columns)
