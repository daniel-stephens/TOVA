# Docker deployment

## Check if API serves requets

```bash
curl -i http://localhost:11000/docs 
```

```bash
curl -i http://localhost:11000/health
```

## Check if the UI container sees the API service

```bash
docker compose exec web sh -lc 'getent hosts api; printenv API_BASE_URL; curl -sf http://api:11000/health || curl -i http://api:11000/docs'
```