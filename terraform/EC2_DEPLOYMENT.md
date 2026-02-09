# EC2 Deployment with Terraform

This guide shows you how to deploy TOVA on EC2 using Terraform with automatic installation via shell script.

## Overview

The EC2 Terraform module automatically:
- Creates EC2 instance with proper security groups
- Installs Docker, Docker Compose, Git, and all dependencies
- Clones your repository (or prepares directory structure)
- Creates `.env` file with database credentials
- Builds and starts all Docker containers
- Sets up auto-start on reboot

## Prerequisites

1. **AWS CLI** installed and configured
2. **Terraform** >= 1.0 installed
3. **Git repository** (optional - you can upload code manually)

## Step 1: EC2 Key Pair (Optional)

You have two options:

### Option A: Let Terraform Create It (Recommended)

Terraform can automatically create a key pair for you. Just set:

```hcl
ec2_create_key_pair = true
ec2_key_pair_name = ""  # Not needed
```

The private key will be saved to `terraform/{project}-{environment}-key.pem`

### Option B: Use Existing Key Pair

If you already have a key pair:

1. Go to AWS Console → EC2 → Key Pairs
2. Note the name of your existing key pair
3. In `terraform.tfvars`:
   ```hcl
   ec2_create_key_pair = false
   ec2_key_pair_name = "your-existing-key-name"
   ```

See `KEY_PAIR_GUIDE.md` for detailed information.

## Step 2: Configure Terraform Variables

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:

```hcl
aws_region   = "us-east-1"
project_name = "tova"
environment  = "dev"

# VPC Configuration
vpc_cidr = "10.0.0.0/16"

# Database Configuration
db_instance_class   = "db.t3.micro"
db_allocated_storage = 20
db_name             = "mydb"
db_username         = "tova_admin"
db_password         = "YourStrongPassword123!"  # Change this!

# Application Configuration
openai_api_key = "sk-..."  # Optional

# Deployment Type - Set to "ec2"
deployment_type = "ec2"

# EC2 Configuration
ec2_instance_type    = "t3.medium"
ec2_create_key_pair  = true  # Set to true to create automatically, false to use existing
ec2_key_pair_name    = ""  # Only needed if ec2_create_key_pair = false
ec2_allowed_ssh_cidrs = ["YOUR_IP/32"]  # e.g., ["203.0.113.0/32"]
ec2_use_rds          = true  # Use RDS (recommended)
ec2_repo_url         = "https://github.com/your-org/TOVA.git"  # Optional
ec2_repo_branch      = "main"
ec2_enable_solr      = true
```

**Important**: 
- Replace `YOUR_IP/32` with your actual IP address (find it at https://whatismyipaddress.com/)
- If using `ec2_create_key_pair = true`, Terraform will create the key pair automatically
- If using existing key pair, set `ec2_create_key_pair = false` and provide `ec2_key_pair_name`
- Set a strong `db_password`

## Step 3: Initialize Terraform

```bash
terraform init
```

This will download the AWS provider and the random provider.

## Step 4: Review Deployment Plan

```bash
terraform plan
```

Review what will be created:
- VPC and networking
- RDS PostgreSQL database
- EC2 instance
- Security groups
- Elastic IP

## Step 5: Deploy

```bash
terraform apply
```

Type `yes` when prompted. This will take about 5-10 minutes.

## Step 6: Get Connection Information

After deployment completes:

```bash
# Get EC2 public IP
terraform output ec2_public_ip

# Get private key path (if created by Terraform)
terraform output ec2_private_key_path

# Get SSH command
terraform output ec2_ssh_command

# Get Web URL
terraform output web_url

# Get API URL
terraform output api_url
```

### Using the Private Key

If Terraform created the key pair:

```bash
# Get the key file path
KEY_FILE=$(terraform output -raw ec2_private_key_path)
PUBLIC_IP=$(terraform output -raw ec2_public_ip)

# SSH into instance
ssh -i $KEY_FILE ec2-user@$PUBLIC_IP
```

## Step 7: Access Your Application

### Wait for Setup to Complete

The user data script runs automatically on first boot. Wait 3-5 minutes, then:

1. **Check setup logs**:
   ```bash
   ssh -i your-key.pem ec2-user@<public_ip>
   sudo tail -f /var/log/tova-setup.log
   ```

2. **Check service status**:
   ```bash
   docker compose ps
   ```

3. **View logs**:
   ```bash
   docker compose logs -f
   ```

### Access URLs

- **Web UI**: `http://<public_ip>:8080`
- **API**: `http://<public_ip>:8000`
- **API Health**: `http://<public_ip>:8000/health`

## Troubleshooting

### Services Not Starting

```bash
# SSH into instance
ssh -i your-key.pem ec2-user@<public_ip>

# Check Docker
docker ps
docker compose ps

# Check logs
docker compose logs api
docker compose logs web

# Restart services
cd ~/TOVA
docker compose restart
```

### Setup Script Failed

```bash
# View setup log
sudo cat /var/log/tova-setup.log

# Manually run setup
cd ~/TOVA
./start-tova.sh
```

### Can't Access from Browser

1. Check security group allows HTTP (port 80) or your app port (8080)
2. Check services are running: `docker compose ps`
3. Check firewall on instance

### Repository Clone Failed

If you didn't provide `ec2_repo_url`, you can manually upload code:

```bash
# On your local machine
tar -czf tova-app.tar.gz TOVA/

# Upload to EC2
scp -i your-key.pem tova-app.tar.gz ec2-user@<public_ip>:~/

# On EC2
cd ~
tar -xzf tova-app.tar.gz
cd TOVA
docker compose up -d
```

## Manual Steps (If Needed)

If the automatic setup didn't complete:

```bash
# SSH into instance
ssh -i your-key.pem ec2-user@<public_ip>

# Navigate to app directory
cd ~/TOVA

# Build base images
make build-builder
make build-assets

# Start services
docker compose up -d

# Check status
docker compose ps
```

## Updating the Application

### Option 1: Git Repository

If you provided `ec2_repo_url`:

```bash
ssh -i your-key.pem ec2-user@<public_ip>
cd ~/TOVA
git pull
docker compose build
docker compose up -d
```

### Option 2: Manual Upload

```bash
# On local machine
tar -czf tova-app.tar.gz TOVA/
scp -i your-key.pem tova-app.tar.gz ec2-user@<public_ip>:~/

# On EC2
cd ~
rm -rf TOVA
tar -xzf tova-app.tar.gz
cd TOVA
docker compose build
docker compose up -d
```

## Cost Estimate

**Monthly Costs (us-east-1):**
- EC2 t3.medium: ~$30
- RDS db.t3.micro: ~$15
- EBS Storage (30GB): ~$3
- **Total: ~$48/month**

## Security Best Practices

1. ✅ **Change SSH CIDR**: Set `ec2_allowed_ssh_cidrs` to your IP only
2. ✅ **Strong Passwords**: Use strong database password
3. ✅ **Key Pair Security**: Keep your `.pem` file secure
4. ✅ **Regular Updates**: SSH in and run `sudo dnf update`
5. ✅ **SSL/HTTPS**: Set up Let's Encrypt for production

## Cleanup

To remove all resources:

```bash
terraform destroy
```

⚠️ **Warning**: This deletes everything including the database! Make sure you have backups.

## Next Steps

1. ✅ Set up domain name (Route 53)
2. ✅ Configure SSL/HTTPS (Let's Encrypt)
3. ✅ Set up automated backups
4. ✅ Configure CloudWatch monitoring
5. ✅ Set up log aggregation

## Support

For issues:
- Check setup logs: `/var/log/tova-setup.log`
- Check Docker logs: `docker compose logs -f`
- Review troubleshooting section above

