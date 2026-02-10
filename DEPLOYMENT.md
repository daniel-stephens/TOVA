# TOVA Deployment Guide with Gunicorn and Nginx

This guide walks you through deploying the TOVA Flask application using Gunicorn as the WSGI server and Nginx as the reverse proxy.

## Prerequisites

- Ubuntu/Debian server (or similar Linux distribution)
- Python 3.8+ installed
- PostgreSQL installed and running
- Nginx installed
- Root or sudo access

## Step 1: Install Dependencies

### Install system packages

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv nginx postgresql postgresql-contrib
```

### Install Python dependencies

```bash
cd /path/to/TOVA/ui
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 2: Configure Environment Variables

Create a `.env` file or set environment variables:

```bash
export FLASK_SECRET_KEY="your-secret-key-here-change-this"
export DATABASE_URL="postgresql+psycopg2://user:password@localhost:5432/mydb"
export POSTGRES_HOST="localhost"
export POSTGRES_PORT="5432"
export POSTGRES_USER="your_db_user"
export POSTGRES_PASSWORD="your_db_password"
export POSTGRES_DB="your_db_name"
export GUNICORN_BIND="0.0.0.0:8000"
export GUNICORN_WORKERS="4"
export RUN_DB_INIT_ON_START="1"
```

For production, consider using a secrets management system or environment file that's not committed to git.

## Step 3: Configure Gunicorn

The Gunicorn configuration is in `ui/gunicorn_config.py`. You can customize it by setting environment variables:

- `GUNICORN_BIND`: Address and port to bind to (default: `0.0.0.0:8000`)
- `GUNICORN_WORKERS`: Number of worker processes (default: CPU count * 2 + 1)
- `GUNICORN_WORKER_CLASS`: Worker class (default: `sync`)
- `GUNICORN_LOG_LEVEL`: Log level (default: `info`)
- `GUNICORN_ACCESS_LOG`: Access log file path (default: stdout)
- `GUNICORN_ERROR_LOG`: Error log file path (default: stderr)

### Test Gunicorn manually

```bash
cd /path/to/TOVA/ui
source venv/bin/activate
gunicorn --config gunicorn_config.py app:server
```

Visit `http://localhost:8000` to verify it's working. Press Ctrl+C to stop.

## Step 4: Configure Nginx

### Update the Nginx configuration

1. Edit `nginx/tova.conf` and update:
   - `server_name` with your domain name
   - Static file paths to match your deployment
   - SSL certificates if using HTTPS

2. Copy the configuration to Nginx:

```bash
sudo cp nginx/tova.conf /etc/nginx/sites-available/tova
sudo ln -s /etc/nginx/sites-available/tova /etc/nginx/sites-enabled/
```

3. Remove the default site (optional):

```bash
sudo rm /etc/nginx/sites-enabled/default
```

4. Test the Nginx configuration:

```bash
sudo nginx -t
```

5. Create log directory:

```bash
sudo mkdir -p /var/log/nginx
```

6. Reload Nginx:

```bash
sudo systemctl reload nginx
```

## Step 5: Set Up Systemd Service

### Update the systemd service file

Edit `systemd/tova-ui.service` and update:
- `User` and `Group` (typically `www-data` or `nginx`)
- `WorkingDirectory` path
- `PATH` in Environment
- All environment variables
- `ExecStart` path to gunicorn

### Install the service

```bash
sudo cp systemd/tova-ui.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable tova-ui
sudo systemctl start tova-ui
```

### Check service status

```bash
sudo systemctl status tova-ui
```

### View logs

```bash
# Service logs
sudo journalctl -u tova-ui -f

# Nginx logs
sudo tail -f /var/log/nginx/tova_access.log
sudo tail -f /var/log/nginx/tova_error.log
```

## Step 6: Firewall Configuration

If using a firewall, allow HTTP and HTTPS traffic:

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw status
```

## Step 7: SSL/HTTPS Setup (Recommended)

For production, set up SSL certificates using Let's Encrypt:

```bash
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com -d www.your-domain.com
```

Certbot will automatically update your Nginx configuration.

## Common Commands

### Service Management

```bash
# Start service
sudo systemctl start tova-ui

# Stop service
sudo systemctl stop tova-ui

# Restart service
sudo systemctl restart tova-ui

# Reload service (graceful restart)
sudo systemctl reload tova-ui

# Check status
sudo systemctl status tova-ui

# View logs
sudo journalctl -u tova-ui -f
```

### Nginx Management

```bash
# Test configuration
sudo nginx -t

# Reload configuration
sudo systemctl reload nginx

# Restart Nginx
sudo systemctl restart nginx

# View access logs
sudo tail -f /var/log/nginx/tova_access.log

# View error logs
sudo tail -f /var/log/nginx/tova_error.log
```

### Gunicorn Management

```bash
# Manual start (for testing)
cd /path/to/TOVA/ui
source venv/bin/activate
gunicorn --config gunicorn_config.py app:server

# Graceful restart (sends HUP signal)
sudo systemctl reload tova-ui
```

## Troubleshooting

### Service won't start

1. Check service status: `sudo systemctl status tova-ui`
2. Check logs: `sudo journalctl -u tova-ui -n 50`
3. Verify paths in service file are correct
4. Ensure virtual environment exists and has dependencies installed
5. Check database connection settings

### 502 Bad Gateway

1. Verify Gunicorn is running: `sudo systemctl status tova-ui`
2. Check Gunicorn is listening on the correct port: `sudo netstat -tlnp | grep 8000`
3. Verify Nginx upstream configuration matches Gunicorn bind address
4. Check Gunicorn logs: `sudo journalctl -u tova-ui -f`

### Static files not loading

1. Verify static file path in Nginx config matches your deployment
2. Check file permissions: `sudo chown -R www-data:www-data /path/to/static`
3. Verify Nginx can read the directory: `sudo ls -la /path/to/static`

### Database connection errors

1. Verify PostgreSQL is running: `sudo systemctl status postgresql`
2. Check database credentials in environment variables
3. Test database connection: `psql -h localhost -U your_user -d your_db`
4. Ensure database exists and user has proper permissions

## Performance Tuning

### Gunicorn Workers

Adjust the number of workers based on your server:
- Formula: `(2 × CPU cores) + 1`
- For I/O-bound apps, consider using `gevent` worker class
- Monitor memory usage and adjust accordingly

### Nginx

- Enable gzip compression in Nginx config
- Adjust `client_max_body_size` for file uploads
- Consider using Nginx caching for static assets

## Security Considerations

1. **Change default secret key**: Set a strong `FLASK_SECRET_KEY`
2. **Use HTTPS**: Always use SSL/TLS in production
3. **Firewall**: Only expose necessary ports (80, 443)
4. **Database**: Use strong passwords and limit database user permissions
5. **File permissions**: Ensure proper ownership and permissions on files
6. **Environment variables**: Never commit secrets to version control
7. **Regular updates**: Keep system packages and Python dependencies updated

## Monitoring

Consider setting up monitoring for:
- Service uptime
- Response times
- Error rates
- Resource usage (CPU, memory, disk)
- Database connection pool

Tools like Prometheus, Grafana, or simple health check endpoints can help.

## Backup

Regularly backup:
- Database (PostgreSQL dumps)
- Application code
- Configuration files
- Static files and uploads

## Rollback Procedure

If you need to rollback:

```bash
# Stop service
sudo systemctl stop tova-ui

# Restore previous code/configuration
git checkout <previous-commit>
# or restore from backup

# Restart service
sudo systemctl start tova-ui
```

