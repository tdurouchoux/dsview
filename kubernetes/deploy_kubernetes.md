# Deploy on kubernetes 

> Next step would be to make it a helm charts

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

2. Generate passwords encodings
```
htpasswd -c auth_label <username>
htpasswd -c auth_ingest <username>
```

3. Save passwords as secrets 
```
kubectl create secret generic basic-auth-label --from-file=auth_label
kubectl create secret generic basic-auth-ingest --from-files=auth_ingest
```

## Step 3 : Apply kubernetes configuration

```
kubectl apply -f kube_full_deploy.yaml
```