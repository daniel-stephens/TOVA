variable "project_name" {
  description = "Project name"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.medium"
}

variable "ami_id" {
  description = "AMI ID (leave empty to use latest Ubuntu 22.04 LTS)"
  type        = string
  default     = ""
}

variable "key_pair_name" {
  description = "Name of the EC2 Key Pair (must exist in AWS)"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "subnet_id" {
  description = "Subnet ID for EC2 instance"
  type        = string
}

variable "volume_size" {
  description = "Root volume size in GB"
  type        = number
  default     = 100
}

variable "allowed_ssh_cidrs" {
  description = "CIDR blocks allowed to SSH"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # ⚠️ Change this!
}

variable "use_rds" {
  description = "Use RDS PostgreSQL instead of Docker PostgreSQL"
  type        = bool
  default     = false
}

variable "rds_endpoint" {
  description = "RDS endpoint (required if use_rds = true)"
  type        = string
  default     = ""
}

variable "postgres_user" {
  description = "PostgreSQL username"
  type        = string
}

variable "postgres_password" {
  description = "PostgreSQL password"
  type        = string
  sensitive   = true
}

variable "postgres_db" {
  description = "PostgreSQL database name"
  type        = string
}

variable "openai_api_key" {
  description = "OpenAI API key"
  type        = string
  default     = ""
  sensitive   = true
}

variable "repo_url" {
  description = "Git repository URL to clone (optional)"
  type        = string
  default     = ""
}

variable "repo_branch" {
  description = "Git repository branch"
  type        = string
  default     = "main"
}

variable "enable_solr" {
  description = "Enable Solr services"
  type        = bool
  default     = true
}

variable "ec2_allow_public_web" {
  description = "Allow public access to Web UI on port 8080 (if false, only allowed_ssh_cidrs can access)"
  type        = bool
  default     = true
}

variable "ec2_allow_public_api" {
  description = "Allow public access to API on port 8000 (if false, only allowed_ssh_cidrs can access)"
  type        = bool
  default     = true
}

