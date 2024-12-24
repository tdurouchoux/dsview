

## Step 1 : Set dsview secrets from .env

```
kubectl create secret generic dsview-secret --from-env-file=.env
```

## Step 2 : Generate password for streamlit and api 

1. install htpasswd
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