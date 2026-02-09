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

variable "private_subnet_ids" {
  description = "Private subnet IDs for EFS mount targets"
  type        = list(string)
}

variable "allowed_security_group_ids" {
  description = "Security group IDs allowed to access EFS"
  type        = list(string)
  default     = []
}

variable "vpc_cidr" {
  description = "VPC CIDR block for allowing NFS access (used if no security groups provided)"
  type        = string
  default     = "10.0.0.0/16"
}

