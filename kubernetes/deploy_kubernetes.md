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
kubectl apply -f dsview_ingress.yaml
```

## Deployment removal :

```
kubectl delete deployment dsview
kubectl delete svc dsview-service
kubectl delete ingress dsview-ingest-ingress
kubectl delete ingress dsview-dashboard-ingress
kubectl delete ingress dsview-labels-ingress
```


## Postgres 

- initialize database 
- Create schemas