#!/bin/bash
set -e
exec > >(tee /var/log/tova-setup.log) 2>&1

. /etc/os-release 2>/dev/null || exit 1
U=$(id ec2-user &>/dev/null && echo ec2-user || echo ubuntu)
H="/home/$U"

# Install Docker
if [ "$ID" = "amzn" ]; then
  dnf update -y && dnf install -y docker && systemctl start docker && systemctl enable docker
  usermod -aG docker $U
elif [ "$ID" = "ubuntu" ]; then
  apt update -y && apt install -y docker.io && systemctl start docker && systemctl enable docker
  usermod -aG docker $U
fi

# Install Docker Compose
[ -f /usr/local/bin/docker-compose ] || curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose && chmod +x /usr/local/bin/docker-compose

# Install Git and tools
[ "$ID" = "amzn" ] && dnf install -y git make curl || apt install -y git make curl

# Clone repo
[ -n "${repo_url}" ] && sudo -u $U git clone -b ${repo_branch} ${repo_url} $H/TOVA || sudo -u $U mkdir -p $H/TOVA
cd $H/TOVA

# Create .env
sudo -u $U cat > .env <<EOF
VERSION=latest
ASSETS_DATE=$$(date +%Y%m%d)
POSTGRES_USER=${postgres_user}
POSTGRES_PASSWORD=${postgres_password}
POSTGRES_DB=${postgres_db}
DATABASE_URL=postgresql://${postgres_user}:${postgres_password}@${postgres_host}:5432/${postgres_db}
OPENAI_API_KEY=${openai_api_key}
FLASK_SECRET_KEY=${flask_secret_key}
API_BASE_URL=http://localhost:8000
EOF

# Create dirs
sudo -u $U mkdir -p data/{drafts,logs,models} db/data/{solr,zoo}
chown -R $U:$U data db 2>/dev/null || true

# Create startup script (try "docker compose" first, fallback to "docker-compose")
cat > /tmp/st.sh <<'EOS'
#!/bin/bash
set -e
if docker compose version &>/dev/null; then DC="docker compose"; else DC="docker-compose"; fi
cd HD/TOVA
for i in {1..30}; do sudo docker info >/dev/null 2>&1 && break || sleep 2; done
export $$(grep -v '^#' .env | xargs)
echo "Building base images..."
sudo docker build --target builder -t tova-builder:$${VERSION:-latest} -f docker/Dockerfile.base . || exit 1
sudo docker build --target assets -t tova-assets:$${ASSETS_DATE:-$$(date +%Y%m%d)} -f docker/Dockerfile.base . || exit 1
echo "Base images built. Starting services..."
if [ "ES" = "true" ]; then
  sudo $DC build --no-cache api web solr-api || sudo $DC build api web solr-api
  sudo $DC up -d api web postgres solr-api solr zoo solr_config || sudo $DC up -d
else
  sudo $DC build --no-cache api web || sudo $DC build api web
  sudo $DC up -d api web postgres || sudo $DC up -d
fi
EOS
sed -i "s|HD|$H|g; s|ES|${enable_solr}|g" /tmp/st.sh
mv /tmp/st.sh $H/start-tova.sh
chmod +x $H/start-tova.sh && chown $U:$U $H/start-tova.sh

# Create systemd service (try "docker compose" first, fallback to "docker-compose")
cat > /etc/systemd/system/tova.service <<EOF
[Unit]
Description=TOVA
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$H/TOVA
ExecStart=/bin/sh -c 'docker compose up -d 2>/dev/null || docker-compose up -d'
ExecStop=/bin/sh -c 'docker compose down 2>/dev/null || docker-compose down'
User=$U

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable tova.service

# Start in background
sudo -u $U bash -c "nohup bash -c 'sleep 60 && $H/start-tova.sh' > $H/start.log 2>&1 &" || nohup bash -c "sleep 60 && $H/start-tova.sh" > $H/start.log 2>&1 &
