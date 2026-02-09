variable "project_name" {
  description = "Project name"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "public_subnet_ids" {
  description = "Public subnet IDs for ALB"
  type        = list(string)
}

variable "private_subnet_ids" {
  description = "Private subnet IDs for ECS tasks"
  type        = list(string)
}

variable "api_repo_url" {
  description = "ECR repository URL for API"
  type        = string
}

variable "web_repo_url" {
  description = "ECR repository URL for Web"
  type        = string
}

variable "solr_api_repo_url" {
  description = "ECR repository URL for Solr API"
  type        = string
}

variable "solr_repo_url" {
  description = "ECR repository URL for Solr"
  type        = string
}

variable "database_url" {
  description = "Database connection URL"
  type        = string
  sensitive   = true
}

variable "openai_api_key" {
  description = "OpenAI API key (optional)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "alb_certificate_arn" {
  description = "ARN of SSL certificate for ALB"
  type        = string
  default     = ""
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = ""
}

variable "efs_file_system_id" {
  description = "EFS file system ID for persistent storage (required)"
  type        = string
}

