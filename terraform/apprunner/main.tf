# AWS App Runner Deployment - Simpler alternative to ECS
# App Runner automatically handles container orchestration, scaling, and load balancing

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
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

# ECR Repositories (if not already created)
resource "aws_ecr_repository" "api" {
  name                 = "${var.project_name}-${var.environment}-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "web" {
  name                 = "${var.project_name}-${var.environment}-web"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# App Runner Service for API
resource "aws_apprunner_service" "api" {
  service_name = "${var.project_name}-${var.environment}-api"

  source_configuration {
    image_repository {
      image_configuration {
        port = "8000"
        runtime_environment_variables = {
          PYTHONUNBUFFERED = "1"
          DATABASE_URL     = var.database_url
          DRAFTS_SAVE      = "/data/drafts"
          API_SOLR_URL     = "http://${aws_apprunner_service.solr_api.service_url}:8001"
        }
      }
      image_identifier      = "${aws_ecr_repository.api.repository_url}:latest"
      image_repository_type = "ECR"
    }
    auto_deployments_enabled = false
  }

  instance_configuration {
    cpu               = "1 vCPU"
    memory            = "2 GB"
    instance_role_arn = aws_iam_role.apprunner_instance.arn
  }

  health_check_configuration {
    protocol            = "HTTP"
    path                = "/health"
    interval            = 10
    timeout             = 5
    healthy_threshold   = 1
    unhealthy_threshold = 5
  }

  network_configuration {
    egress_configuration {
      egress_type = "VPC"
      vpc_connector_arn = aws_apprunner_vpc_connector.main.arn
    }
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-api"
  }
}

# App Runner Service for Web UI
resource "aws_apprunner_service" "web" {
  service_name = "${var.project_name}-${var.environment}-web"

  source_configuration {
    image_repository {
      image_configuration {
        port = "8080"
        runtime_environment_variables = {
          FLASK_DEBUG  = var.environment == "prod" ? "0" : "1"
          API_BASE_URL = "https://${aws_apprunner_service.api.service_url}/api"
          DATABASE_URL = var.database_url
        }
      }
      image_identifier      = "${aws_ecr_repository.web.repository_url}:latest"
      image_repository_type = "ECR"
    }
    auto_deployments_enabled = false
  }

  instance_configuration {
    cpu               = "1 vCPU"
    memory            = "2 GB"
    instance_role_arn = aws_iam_role.apprunner_instance.arn
  }

  health_check_configuration {
    protocol            = "HTTP"
    path                = "/"
    interval            = 10
    timeout             = 5
    healthy_threshold   = 1
    unhealthy_threshold = 5
  }

  network_configuration {
    egress_configuration {
      egress_type       = "VPC"
      vpc_connector_arn = aws_apprunner_vpc_connector.main.arn
    }
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-web"
  }
}

# Note: Solr and Zookeeper are better deployed via ECS or managed service
# App Runner is best for stateless web services

# VPC Connector for App Runner to access RDS and other VPC resources
# Note: Requires at least 2 subnets in different AZs
resource "aws_apprunner_vpc_connector" "main" {
  vpc_connector_name = "${var.project_name}-${var.environment}-vpc-connector"
  subnets            = length(var.private_subnet_ids) >= 2 ? slice(var.private_subnet_ids, 0, 2) : var.private_subnet_ids
  security_groups    = [aws_security_group.apprunner.id]
}

# Security Group for App Runner
resource "aws_security_group" "apprunner" {
  name_prefix = "${var.project_name}-${var.environment}-apprunner-"
  vpc_id      = var.vpc_id
  description = "Security group for App Runner services"

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-apprunner-sg"
  }
}

# IAM Role for App Runner Instance
resource "aws_iam_role" "apprunner_instance" {
  name = "${var.project_name}-${var.environment}-apprunner-instance"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "build.apprunner.amazonaws.com"
      }
    }]
  })
}

# IAM Role for App Runner Service
resource "aws_iam_role" "apprunner_service" {
  name = "${var.project_name}-${var.environment}-apprunner-service"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "build.apprunner.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "apprunner_service" {
  role       = aws_iam_role.apprunner_service.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicy"
}

# RDS PostgreSQL (same as ECS deployment)
module "rds" {
  source = "../modules/rds"
  
  project_name        = var.project_name
  environment         = var.environment
  vpc_id              = var.vpc_id
  private_subnet_ids  = var.private_subnet_ids
  db_instance_class   = var.db_instance_class
  db_allocated_storage = var.db_allocated_storage
  db_name             = var.db_name
  db_username         = var.db_username
  db_password         = var.db_password
  allowed_cidr_blocks = [var.vpc_cidr]
}

# Security Group for RDS (allow App Runner)
resource "aws_security_group_rule" "rds_from_apprunner" {
  type                     = "ingress"
  from_port                = 5432
  to_port                  = 5432
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.apprunner.id
  security_group_id        = module.rds.db_security_group_id
  description              = "Allow App Runner to access RDS"
}

