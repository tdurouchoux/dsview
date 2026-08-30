# Deploy on kubernetes

> Next step would be to make it a helm charts

## Step 0 : Create a service account

```
kubectl create serviceaccount dsview-sa
```

## Step 1 : Set dsview secrets from .env

```
kubectl create secret generic dsview-secret --from-env-file=.env
```

## Step 2 : Generate password for streamlit and api

1. Install `htpasswd`

```
sudo apt-get update
sudo apt-get install apache2-utils
```

2. Generate passwords for label interface

```
htpasswd -c auth <username>
kubectl create secret generic basic-auth-dsview-dashboard --from-file=auth
rm auth
```

3. Generate passwords for ingest API

```
htpasswd -c auth <username>
kubectl create secret generic basic-auth-dsview-ingest --from-file=auth
rm auth
```

4. Get header for ingest api

```
echo -n "<username>:<password>" | base64
```

Header should like : `Authorization: Basic <base64_encoded_auth>`

## Step 3 : Apply kubernetes configuration

```
kubectl apply -f dsview_deployment.yaml
kubectl apply -f dsview_digest_cronjob.yaml
```

> The digest CronJob reads `DIGEST_SMTP_USER`, `DIGEST_SMTP_PASSWORD` and
> `DIGEST_EMAIL_TO` from `dsview-secret` - add them to `.env` (see
> `.env.example`) before Step 1.

## Deployment removal :

```
kubectl delete -f dsview_deployment.yaml
kubectl delete -f dsview_digest_cronjob.yaml
```

> option `--ignore-not-found=true`

## Postgres

- initialize database
- Create schemas
