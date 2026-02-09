variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "tova"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

# Database variables
variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "db_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 20
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "mydb"
}

variable "db_username" {
  description = "Database master username (cannot be 'admin', 'root', 'postgres', etc.)"
  type        = string
  default     = "tova_admin"
}

variable "db_password" {
  description = "Database master password"
  type        = string
  sensitive   = true
}

# Application variables
variable "openai_api_key" {
  description = "OpenAI API key (optional)"
  type        = string
  default     = ""
  sensitive   = true
}

# Domain and SSL
variable "domain_name" {
  description = "Domain name for the application (optional)"
  type        = string
  default     = ""
}

variable "alb_certificate_arn" {
  description = "ARN of SSL certificate for ALB (required if domain_name is set)"
  type        = string
  default     = ""
}

variable "force_delete_ecr" {
  description = "Force delete ECR repositories on destroy (deletes all images - use with caution!)"
  type        = bool
  default     = false
}

# Deployment type
variable "deployment_type" {
  description = "Deployment type: 'ecs' or 'ec2'"
  type        = string
  default     = "ecs"
  validation {
    condition     = contains(["ecs", "ec2"], var.deployment_type)
    error_message = "Deployment type must be either 'ecs' or 'ec2'."
  }
}

# EC2 variables
variable "ec2_instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.medium"
}

variable "ec2_create_key_pair" {
  description = "Create a new EC2 Key Pair in Terraform (if false, use existing key_pair_name)"
  type        = bool
  default     = true
}

variable "ec2_key_pair_name" {
  description = "Name of existing EC2 Key Pair (only used if ec2_create_key_pair = false)"
  type        = string
  default     = ""
}

variable "ec2_allowed_ssh_cidrs" {
  description = "CIDR blocks allowed to SSH to EC2"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # ⚠️ Change this to your IP for security!
}

variable "ec2_use_rds" {
  description = "Use RDS PostgreSQL instead of Docker PostgreSQL on EC2"
  type        = bool
  default     = true
}

variable "ec2_repo_url" {
  description = "Git repository URL for EC2 deployment (optional - leave empty to manually upload code)"
  type        = string
  default     = ""
}

variable "ec2_repo_branch" {
  description = "Git repository branch"
  type        = string
  default     = "main"
}

variable "ec2_enable_solr" {
  description = "Enable Solr services on EC2"
  type        = bool
  default     = true
}

