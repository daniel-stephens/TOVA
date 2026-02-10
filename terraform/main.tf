terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.0"
    }
  }

  # Uncomment and configure if you want to use remote state
  # backend "s3" {
  #   bucket = "tova-terraform-state"
  #   key    = "terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "TOVA"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Generate TLS private key for EC2 Key Pair (if create_key_pair = true)
resource "tls_private_key" "ec2_key" {
  count     = var.deployment_type == "ec2" && var.ec2_create_key_pair ? 1 : 0
  algorithm = "RSA"
  rsa_bits  = 4096
}

# Create AWS Key Pair from generated public key
resource "aws_key_pair" "tova" {
  count      = var.deployment_type == "ec2" && var.ec2_create_key_pair ? 1 : 0
  key_name   = "${var.project_name}-${var.environment}-key"
  public_key = tls_private_key.ec2_key[0].public_key_openssh

  tags = {
    Name = "${var.project_name}-${var.environment}-key"
  }
}

# Save private key to file (only if created by Terraform)
resource "local_file" "private_key" {
  count           = var.deployment_type == "ec2" && var.ec2_create_key_pair ? 1 : 0
  content         = tls_private_key.ec2_key[0].private_key_pem
  filename        = "${path.module}/${var.project_name}-${var.environment}-key.pem"
  file_permission = "0400"
}

# VPC Module
module "vpc" {
  source = "./modules/vpc"
  
  project_name     = var.project_name
  environment      = var.environment
  vpc_cidr         = var.vpc_cidr
  availability_zones = data.aws_availability_zones.available.names
}

# ECR Repositories
module "ecr" {
  source = "./modules/ecr"
  
  project_name = var.project_name
  repositories = ["api", "web", "solr-api", "solr", "base"]
  force_delete = var.force_delete_ecr  # Set to true in terraform.tfvars to allow deletion
}

# RDS PostgreSQL
module "rds" {
  source = "./modules/rds"
  
  project_name        = var.project_name
  environment         = var.environment
  vpc_id              = module.vpc.vpc_id
  private_subnet_ids  = module.vpc.private_subnet_ids
  db_instance_class   = var.db_instance_class
  db_allocated_storage = var.db_allocated_storage
  db_name             = var.db_name
  db_username         = var.db_username
  db_password         = var.db_password
  allowed_cidr_blocks = [var.vpc_cidr]
}

# EFS for persistent storage (Solr and Zookeeper data)
module "efs" {
  source = "./modules/efs"
  
  project_name              = var.project_name
  environment              = var.environment
  vpc_id                   = module.vpc.vpc_id
  vpc_cidr                 = var.vpc_cidr
  private_subnet_ids       = module.vpc.private_subnet_ids
  allowed_security_group_ids = [] # Allow from entire VPC for simplicity
}

# ECS Cluster (comment out if using EC2)
module "ecs" {
  count  = var.deployment_type == "ecs" ? 1 : 0
  source = "./modules/ecs"
  
  project_name       = var.project_name
  environment        = var.environment
  vpc_id             = module.vpc.vpc_id
  public_subnet_ids  = module.vpc.public_subnet_ids
  private_subnet_ids = module.vpc.private_subnet_ids
  
  # ECR repositories
  api_repo_url      = module.ecr.repository_urls["api"]
  web_repo_url      = module.ecr.repository_urls["web"]
  solr_api_repo_url = module.ecr.repository_urls["solr-api"]
  solr_repo_url     = module.ecr.repository_urls["solr"]
  
  # Database connection
  database_url = "postgresql://${var.db_username}:${var.db_password}@${module.rds.db_endpoint}/${var.db_name}"
  
  # EFS for persistent storage
  efs_file_system_id = module.efs.file_system_id
  
  # Environment variables
  openai_api_key = var.openai_api_key
  
  # Load balancer
  alb_certificate_arn = var.alb_certificate_arn
  domain_name         = var.domain_name
}

# EC2 Module (alternative to ECS)
module "ec2" {
  count  = var.deployment_type == "ec2" ? 1 : 0
  source = "./modules/ec2"
  
  project_name  = var.project_name
  environment   = var.environment
  aws_region   = var.aws_region
  instance_type = var.ec2_instance_type
  key_pair_name = var.ec2_create_key_pair ? aws_key_pair.tova[0].key_name : var.ec2_key_pair_name
  
  vpc_id    = module.vpc.vpc_id
  subnet_id = module.vpc.public_subnet_ids[0]
  
  # Use RDS from existing module
  use_rds         = var.ec2_use_rds
  rds_endpoint    = var.ec2_use_rds ? module.rds.db_endpoint : ""
  postgres_user   = var.db_username
  postgres_password = var.db_password
  postgres_db     = var.db_name
  
  openai_api_key = var.openai_api_key
  
  allowed_ssh_cidrs = var.ec2_allowed_ssh_cidrs
  
  # Public access controls
  ec2_allow_public_web = var.ec2_allow_public_web
  ec2_allow_public_api = var.ec2_allow_public_api
  
  # Git repository (optional)
  repo_url   = var.ec2_repo_url
  repo_branch = var.ec2_repo_branch
  
  enable_solr = var.ec2_enable_solr
}

# Outputs
output "vpc_id" {
  value = module.vpc.vpc_id
}

# ECS Outputs
output "alb_dns_name" {
  value     = var.deployment_type == "ecs" && length(module.ecs) > 0 ? module.ecs[0].alb_dns_name : null
}

output "web_url" {
  value = var.deployment_type == "ecs" && length(module.ecs) > 0 ? (
    var.domain_name != "" ? "https://${var.domain_name}" : "http://${module.ecs[0].alb_dns_name}"
  ) : (
    var.deployment_type == "ec2" && length(module.ec2) > 0 ? module.ec2[0].web_url : null
  )
}

output "api_url" {
  value = var.deployment_type == "ecs" && length(module.ecs) > 0 ? (
    var.domain_name != "" ? "https://${var.domain_name}/api" : "http://${module.ecs[0].alb_dns_name}/api"
  ) : (
    var.deployment_type == "ec2" && length(module.ec2) > 0 ? module.ec2[0].api_url : null
  )
}

# EC2 Outputs
output "ec2_public_ip" {
  value     = var.deployment_type == "ec2" && length(module.ec2) > 0 ? module.ec2[0].public_ip : null
}

output "ec2_ssh_command" {
  value     = var.deployment_type == "ec2" && length(module.ec2) > 0 ? module.ec2[0].ssh_command : null
}

output "ec2_instance_id" {
  value     = var.deployment_type == "ec2" && length(module.ec2) > 0 ? module.ec2[0].instance_id : null
}

# Key Pair Outputs
output "ec2_key_pair_name" {
  description = "Name of the EC2 Key Pair (created or existing)"
  value       = var.deployment_type == "ec2" && var.ec2_create_key_pair ? aws_key_pair.tova[0].key_name : var.ec2_key_pair_name
}

output "ec2_private_key_path" {
  description = "Path to the private key file (only if created by Terraform)"
  value       = var.deployment_type == "ec2" && var.ec2_create_key_pair ? local_file.private_key[0].filename : null
  sensitive   = false
}

output "ec2_private_key_save_instructions" {
  description = "Instructions for saving the private key"
  value = var.deployment_type == "ec2" && var.ec2_create_key_pair ? join("\n", [
    "Private key saved to: ${local_file.private_key[0].filename}",
    "Use it to SSH: ssh -i ${local_file.private_key[0].filename} ec2-user@<public_ip>"
  ]) : null
}

output "rds_endpoint" {
  value     = module.rds.db_endpoint
  sensitive = true
}

output "ecr_repository_urls" {
  value = module.ecr.repository_urls
}

