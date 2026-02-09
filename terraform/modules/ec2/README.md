# EC2 Module for TOVA

This Terraform module creates an EC2 instance with Docker and automatically installs and configures TOVA.

## Features

- ✅ Automatically installs Docker and Docker Compose
- ✅ Installs Git and other required tools
- ✅ Clones repository (if provided) or creates directory structure
- ✅ Creates `.env` file with database credentials
- ✅ Sets up all required directories
- ✅ Automatically builds and starts Docker containers
- ✅ Creates systemd service for auto-start on reboot
- ✅ Configures security groups
- ✅ Allocates Elastic IP for static address
- ✅ Supports both RDS and Docker PostgreSQL

## Usage

### Basic Usage

```hcl
module "ec2" {
  source = "./modules/ec2"
  
  project_name  = "tova"
  environment   = "dev"
  aws_region    = "us-east-1"
  instance_type = "t3.medium"
  key_pair_name = "my-key-pair"
  
  vpc_id    = module.vpc.vpc_id
  subnet_id = module.vpc.public_subnet_ids[0]
  
  # Database configuration
  use_rds         = true
  rds_endpoint    = module.rds.db_endpoint
  postgres_user   = "tova_admin"
  postgres_password = "secure-password"
  postgres_db     = "mydb"
  
  openai_api_key = "sk-..."
  allowed_ssh_cidrs = ["203.0.113.0/32"]
  
  # Optional: Git repository
  repo_url   = "https://github.com/your-org/TOVA.git"
  repo_branch = "main"
  
  enable_solr = true
}
```

## Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|----------|
| project_name | Project name | string | - | yes |
| environment | Environment name | string | - | yes |
| aws_region | AWS region | string | - | yes |
| instance_type | EC2 instance type | string | "t3.medium" | no |
| ami_id | AMI ID (empty = latest Amazon Linux 2023) | string | "" | no |
| key_pair_name | EC2 Key Pair name | string | - | yes |
| vpc_id | VPC ID | string | - | yes |
| subnet_id | Subnet ID | string | - | yes |
| volume_size | Root volume size (GB) | number | 30 | no |
| allowed_ssh_cidrs | CIDR blocks for SSH access | list(string) | ["0.0.0.0/0"] | no |
| use_rds | Use RDS PostgreSQL | bool | false | no |
| rds_endpoint | RDS endpoint (if use_rds = true) | string | "" | no |
| postgres_user | PostgreSQL username | string | - | yes |
| postgres_password | PostgreSQL password | string | - | yes |
| postgres_db | PostgreSQL database name | string | - | yes |
| openai_api_key | OpenAI API key | string | "" | no |
| repo_url | Git repository URL | string | "" | no |
| repo_branch | Git repository branch | string | "main" | no |
| enable_solr | Enable Solr services | bool | true | no |

## Outputs

| Name | Description |
|------|-------------|
| instance_id | EC2 instance ID |
| public_ip | Public IP address (Elastic IP) |
| public_dns | Public DNS name |
| ssh_command | SSH command to connect |
| web_url | Web UI URL |
| api_url | API URL |
| security_group_id | Security group ID |

## What Gets Installed

The `user_data.sh` script automatically installs:

1. **Docker** - Container runtime
2. **Docker Compose** - Multi-container orchestration
3. **Git** - Version control
4. **Make, curl, wget, unzip** - Build tools
5. **System updates** - Latest security patches

## Setup Process

1. Instance launches with Amazon Linux 2023
2. User data script runs automatically
3. Installs Docker, Docker Compose, Git
4. Clones repository (if provided) or creates directory
5. Creates `.env` file with database credentials
6. Creates required directories
7. Builds Docker images
8. Starts all services
9. Creates systemd service for auto-start

## Accessing the Application

After deployment:

- **Web UI**: `http://<public_ip>:8080`
- **API**: `http://<public_ip>:8000`
- **SSH**: `ssh -i your-key.pem ec2-user@<public_ip>`

## Troubleshooting

### View Setup Logs

```bash
ssh -i your-key.pem ec2-user@<public_ip>
sudo tail -f /var/log/tova-setup.log
```

### Check Service Status

```bash
docker compose ps
docker compose logs -f
```

### Restart Services

```bash
cd ~/TOVA
docker compose restart
```

### Manual Start

If services didn't start automatically:

```bash
cd ~/TOVA
./start-tova.sh
```

## Security Notes

⚠️ **Important**: Change `allowed_ssh_cidrs` to your IP address:

```hcl
allowed_ssh_cidrs = ["203.0.113.0/32"]  # Your IP only
```

## Cost Estimate

- **t3.medium**: ~$30/month
- **t3.large**: ~$60/month
- **EBS Storage (30GB)**: ~$3/month
- **Elastic IP**: Free (if attached to running instance)

## Next Steps

1. Set up domain name (Route 53)
2. Configure SSL/HTTPS (Let's Encrypt)
3. Set up automated backups
4. Configure CloudWatch monitoring
5. Set up log aggregation

