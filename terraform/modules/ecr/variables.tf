variable "project_name" {
  description = "Project name"
  type        = string
}

variable "repositories" {
  description = "List of ECR repository names"
  type        = list(string)
}

variable "force_delete" {
  description = "Force delete ECR repositories even if they contain images (use with caution!)"
  type        = bool
  default     = false
}

